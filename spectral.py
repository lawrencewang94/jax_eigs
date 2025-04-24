# imports
import jax
import jax.numpy as jnp
import numpy as np
import jax.tree_util as tu

# import refactor.utils as utils
import utils
import jax.flatten_util as fu
'''
https://github.com/google/spectral-density/tree/master/jax
'''
# print("YES jit")

def tree_slice(tree, start_ind, slice_size):
    return jax.tree_map(lambda x: jax.lax.dynamic_slice_in_dim(x, start_ind, slice_size, 0), tree)


# params, batch, det -> loss
def get_loss_wrap(state, loss_fn, bn=True):
    """
    Wraps the loss objective to create a function that computes the loss taking the model and data(batch) as inputs.
    ---
    loss_fn: tx.Loss object;
    ---
    function: (model, batch) -> loss;
    """
    def loss_wrap_bn(params, batch, train=False):
        inputs, targets = batch
        preds = state.apply_fn({'params':params, 'batch_stats':state.batch_stats}, inputs, train=False)
        return loss_fn(preds, batch[1]).mean()

    def loss_wrap(params, batch, train=False):
        inputs, targets = batch
        preds = state.apply_fn({'params': params}, inputs, train=False)
        return loss_fn(preds, batch[1]).mean()

    return loss_wrap_bn if bn else loss_wrap


def hvp_w(w, v, get_loss):
    """
    Computes the HVP from inputs, w, v.
    ---
    loss_wrap: wrapper defined by get_loss_hvp; model: tx.Sequential; batch: data; v: vector;
    ---
    hvp output
    """
    return jax.jvp(jax.grad(get_loss), [w], [v])[1]


@jax.jit
def _tree_add(tree_left, tree_right):
    """Computes tree_left + tree_right."""
    def add(x, y):
        return x + y

    return tu.tree_map(add, tree_left, tree_right)


def _tree_zeros_like(tree):
    def zeros(x):
        return np.zeros_like(x)
    return tu.tree_map(zeros, tree)


def get_hvp_fn(loss_wrap, state, batch, bs=0):
    """
    Returns a function that computes the HVP taking the model and v as inputs
    ---
    loss_wrap: wrapper defined by get_loss_hvp; model: tx.Sequential; batch: data; v: vector;
    ---
    hvp_fn: (model, v) -> hvp
    """
    # get NN structure and count NN params

    num_params = int(utils.count_params(state.params))
    _, model_structure = jax.flatten_util.ravel_pytree(state.params)

    @jax.jit
    def jitted_hvp_w(w, batch, v):
        get_loss = lambda w: loss_wrap(w, batch)
        return hvp_w(w, v, get_loss)

    def hvp_fn_fb(w, v):
        # w is a tree, v is flat
        v = model_structure(v)
        hessian_vp = jitted_hvp_w(w, batch, v)

        hessian_vp_flat, _ = jax.flatten_util.ravel_pytree(hessian_vp)
        return hessian_vp_flat

    def hvp_fn_minib(w, v):
        # w is a tree, v is flat
        v = model_structure(v)
        hessian_vp = _tree_zeros_like(state.params)  # initialise 0 result # EDITE FOR TRANSFORMER

        def body_fun(i, val):
            v, hessian_vp = val
            # batch_data = jax.lax.dynamic_slice_in_dim(batch[0], i*bs, bs)
            # batch_targets = jax.lax.dynamic_slice_in_dim(batch[1], i*bs, bs)
            # partial_vp = jitted_hvp_w(w, (batch_data, batch_targets), v) # edited for TRANSFORMER

            sliced_batch = tuple(tree_slice(arr, i*bs, bs) for arr in batch)
            partial_vp = jitted_hvp_w(w, sliced_batch, v) #
            hessian_vp = _tree_add(hessian_vp, partial_vp)
            return v, hessian_vp

        count = int(len(batch[0])/bs)
        _, hessian_vp = jax.lax.fori_loop(0, count, body_fun, (v, hessian_vp)) # this auto jits (i think)

        hessian_vp_flat, _ = jax.flatten_util.ravel_pytree(hessian_vp)
        hessian_vp_flat /= count
        return hessian_vp_flat

    if bs == 0 or bs >= len(batch[0]):
        return hvp_fn_fb, model_structure, int(num_params)
    else:
        return hvp_fn_minib, model_structure, int(num_params)


# jit when called
def dot_carry(a, b):
    coeff = jnp.dot(a, b)
    a -= coeff * b
    return a, coeff


@jax.jit
def _lanczos_reorthog(w, vecs):
    return jax.lax.scan(dot_carry, w, vecs)


@jax.jit
def _lanczos_w_ops(w, beta, v, v_old):
    w = w - beta * v_old
    alpha = jnp.dot(w, v)
    w = w - alpha * v
    return w, alpha


# EDITED TO INCLUDE RESTARTS
def lanczos_alg(matrix_vector_product, dim, order, rng_key, max_attempts=5):
    # With MPK
    """Lanczos algorithm for tridiagonalizing a real symmetric matrix.

    This function applies Lanczos algorithm of a given order.  This function
    does full reorthogonalization.

    WARNING: This function may take a long time to jit compile (e.g. ~3min for
    order 90 and dim 1e7).

    Args:
    matrix_vector_product: Maps v -> Hv for a real symmetric matrix H.
        Input/Output must be of shape [dim].
    dim: Matrix H is [dim, dim].
    order: An integer corresponding to the number of Lanczos steps to take.
    rng_key: The jax PRNG key.

    Returns:
    tridiag: A tridiagonal matrix of size (order, order).
    vecs: A numpy array of size (order, dim) corresponding to the Lanczos
        vectors.
    """
    count = 0

    while count < max_attempts:
        tridiag = np.zeros((order, order))
        vecs = np.zeros((order, dim))
        rng_key, split = jax.random.split(rng_key)

        init_vec = jax.random.normal(split, shape=(dim,))
        init_vec = init_vec / np.linalg.norm(init_vec)
        vecs[0] = init_vec

        eta_min = 1e-16
        eta_max = jnp.sqrt(0.5)

        beta = 0
        break_flag = False

        for i in range(order):
            v = vecs[i, :].reshape((dim))
            # jax.lax.cond(i==0, lambda x: ( x := jnp.zeros(dim)), lambda x: (x := vecs[i-1, :].reshape((dim))), v_old)
            if i == 0:
                v_old = 0.
            else:
                v_old = vecs[i-1, :].reshape((dim))
            # print("v", v.shape, "\norder", order, "dim", dim)
            w = matrix_vector_product(v)
            # print("w", w.shape, "v", v.shape, "order", order, "dim", dim)

            w, alpha = _lanczos_w_ops(w, beta, v, v_old)
            tridiag[i, i] = alpha

            # Full Reorthogonalization (looks like gram schmidt)
            unorthog_norm = np.linalg.norm(w)
            w_orthog, coeffs = _lanczos_reorthog(w, vecs)
            w_orthog /= unorthog_norm

            eta = np.linalg.norm(w)/unorthog_norm
            # changed from while to if -> twice is enough
            if eta < eta_max:
                if eta < eta_min:
                    count += 1
                    break_flag = True
                    break
                else :
                    # a second reorthog step is needed
                    unorthog_norm2 = np.linalg.norm(w_orthog)
                    w_orthog2, coeffs = _lanczos_reorthog(w_orthog, vecs)
                    w_orthog2 /= unorthog_norm2
                    w_tmp = vecs@w_orthog2
                    eta_min = np.linalg.norm(w_tmp)
                    w = w_orthog2.copy()
            else:
                w = w_orthog.copy()

            beta = jnp.linalg.norm(w)

            if i + 1 < order:
                tridiag[i, i + 1] = (beta)
                tridiag[i + 1, i] = (beta)
                vecs[i + 1] = (w / beta)

        if break_flag:
            continue

        return (tridiag, vecs)

    raise ArithmeticError


def tridiag_to_eigv(tridiag_list):
    """Preprocess the tridiagonal matrices for density estimation.

    Args:
    tridiag_list: Array of shape [num_draws, order, order] List of the
      tridiagonal matrices computed from running num_draws independent runs
      of lanczos. The output of this function can be fed directly into
      eigv_to_density.

    Returns:
    eig_vals: Array of shape [num_draws, order]. The eigenvalues of the
      tridiagonal matricies.
    all_weights: Array of shape [num_draws, order]. The weights associated with
      each eigenvalue. These weights are to be used in the kernel density
      estimate.
    """

    @jax.jit
    def _tridiag_nested(itl):
        nodes, evecs = jnp.linalg.eigh(itl)
        index = jnp.argsort(nodes)
        nodes = nodes[index]
        evecs = evecs[:, index]
        return nodes, evecs

    # Calculating the node / weights from Jacobi matrices.
    num_draws = len(tridiag_list)
    num_lanczos = tridiag_list[0].shape[0]
    eig_vals = np.zeros((num_draws, num_lanczos))
    all_weights = np.zeros((num_draws, num_lanczos))
    all_evecs = np.zeros((num_draws, num_lanczos, num_lanczos))

    for i in range(num_draws):
        nodes, evecs = _tridiag_nested(tridiag_list[i])
        eig_vals[i, :] = nodes
        all_weights[i, :] = evecs[0] ** 2
        all_evecs[i, :] = evecs

    return eig_vals, all_evecs, all_weights


