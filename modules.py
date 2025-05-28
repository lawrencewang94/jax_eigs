import jax
from jax import lax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn  # Linen API
import dataclasses

# import refactor.utils as utils
# import refactor.spectral as spectral
# import refactor.training as training
import utils
import spectral
import training
from typing import Optional, Union
import typing as tp

import jax.flatten_util as fu
import neural_tangents as nt
from scipy.optimize import minimize as glob_opt
from flax.linen.initializers import lecun_normal, he_normal

from scipy.special import softmax

CallableModule = tp.Callable[..., jnp.ndarray]
pi = jnp.pi
exp = jnp.exp

import typing as tp
from optax._src import base, combine, numerics, transform

# Type aliases
DType = np.dtype
ScalarOrSchedule = Union[float, base.Schedule]
import optax_adam_bounded as oab


# python file for loss functions, optimizers, computations, etc.


# @jax.jit
def lognormpdf (x, mean, std):
    log = jnp.log
    # assumes diagonal covariance so 1D mean and 1D cov vectors
    # bounding the std - see deepmind notebook https://github.com/deepmind/neural-processes/blob/master/conditional_neural_process.ipynb
    std = 0.01 + 0.99 * std
    variance = std ** 2
    variation = x - mean
    out = -log(std) - 0.5 * log(2 * pi) - variation ** 2 / (2 * variance)
    return out


def gaussNLL (
        target: jnp.ndarray,
        preds: jnp.ndarray,
) -> jnp.float32:
    # target is target y with the last axis denoting the components
    # pred is mean and cov, shape: bs X n_p X (d+1) X d, with first elem as mean array

    # assume covs are al diag
    target = target.flatten()
    mean = preds[:, :, 0:1].flatten()
    cov = jax.nn.softplus(preds[:, :, 1:2].flatten())

    pdfs = jax.vmap(lognormpdf)(target, mean, cov)
    res = -pdfs

    return res


# Keeping but not refactored to flax/optax yet
def ntk2hess (ntk, norm=True):
    k = ntk.shape[2]
    B = ntk.shape[0]
    assert ntk.shape[0] == ntk.shape[1]
    assert ntk.shape[2] == ntk.shape[3]
    hess = np.transpose(ntk, axes=(0, 2, 1, 3))
    hess = np.reshape(hess, (ntk.shape[0] * ntk.shape[2], ntk.shape[0] * ntk.shape[2]))
    if norm:
        hess /= (k * B)
    return hess


# Keeping but not refactored to flax/optax yet
def compute_empirical_NTK (model, tr_inputs, method=1, hessian=False):
    def model_call (model):
        _, ravel = fu.ravel_pytree(model)

        def call (weights, inputs):
            model_new = ravel(weights)
            return model_new(inputs)

        return call

    kwargs = dict(
        f=model_call(model),
        trace_axes=(),
        vmap_axes=0
    )

    ntk_fn = jax.jit(nt.empirical_ntk_fn(**kwargs, implementation=method))
    x1 = np.array(tr_inputs)

    params, _ = fu.ravel_pytree(model)
    ntk = ntk_fn(x1, None, params)
    hess = ntk2hess(ntk)
    w_ntk, v_ntk = np.linalg.eig(hess)
    if hessian:
        hessian = jax.grad
    # print(w_ntk)
    # print("2/lam0", 2/np.max(w_ntk), "4/lam0", 4/np.max(w_ntk), "12/lam0", 12/np.max(w_ntk))
    return w_ntk


def pool (inputs, init, reduce_fn, window_shape, strides, padding):
    """Helper function to define pooling functions.

    Pooling functions are implemented using the ReduceWindow XLA op.
    NOTE: Be aware that pooling is not generally differentiable.
    That means providing a reduce_fn that is differentiable does not imply that
    pool is differentiable.

    Args:
    inputs: input data with dimensions (batch, window dims..., features).
    init: the initial value for the reduction
    reduce_fn: a reduce function of the form `(T, T) -> T`.
    window_shape: a shape tuple defining the window to reduce over.
    strides: a sequence of `n` integers, representing the inter-window
        strides (default: `(1, ..., 1)`).
    padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
        of `n` `(low, high)` integer pairs that give the padding to apply before
        and after each spatial dimension.
    Returns:
    The output of the reduction for each window slice.
    """
    num_batch_dims = inputs.ndim - (len(window_shape) + 1)
    strides = strides or (1,) * len(window_shape)
    assert len(window_shape) == len(strides), (
        f"len({window_shape}) must equal len({strides})")
    strides = (1,) * num_batch_dims + strides + (1,)
    dims = (1,) * num_batch_dims + window_shape + (1,)
    #     print(strides, dims, num_batch_dims)
    is_single_input = False
    if num_batch_dims == 0:
        # add singleton batch dimension because lax.reduce_window always
        # needs a batch dimension.
        inputs = inputs[None]
        strides = (1,) + strides
        dims = (1,) + dims
        is_single_input = True

    assert inputs.ndim == len(dims), f"len({inputs.shape}) != len({dims})"
    y = lax.reduce_window(inputs, init, reduce_fn, dims, strides, padding)
    if is_single_input:
        y = jnp.squeeze(y, axis=0)
    return y


def max_pool (inputs, window_shape, strides=None, padding="VALID"):
    """Pools the input by taking the maximum of a window slice.

    Args:
    inputs: input data with dimensions (batch, window dims..., features).
    window_shape: a shape tuple defining the window to reduce over.
    strides: a sequence of `n` integers, representing the inter-window
      strides (default: `(1, ..., 1)`).
    padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
      of `n` `(low, high)` integer pairs that give the padding to apply before
      and after each spatial dimension (default: `'VALID'`).
    Returns:
    The maximum for each window slice.
    """
    y = pool(inputs, -jnp.inf, lax.max, window_shape, strides, padding)
    return y


def ave_pool (inputs, window_shape, strides=None, padding="VALID"):
    """Pools the input by taking the maximum of a window slice.

    Args:
    inputs: input data with dimensions (batch, window dims..., features).
    window_shape: a shape tuple defining the window to reduce over.
    strides: a sequence of `n` integers, representing the inter-window
      strides (default: `(1, ..., 1)`).
    padding: either the string `'SAME'`, the string `'VALID'`, or a sequence
      of `n` `(low, high)` integer pairs that give the padding to apply before
      and after each spatial dimension (default: `'VALID'`).
    Returns:
    The maximum for each window slice.
    """
    y = pool(inputs, -jnp.inf, lax.max, window_shape, strides, padding)
    return y


def mean_squared_error (target: jnp.ndarray, preds: jnp.ndarray) -> jnp.ndarray:
    """
    Computes the mean squared error between target and predictions.

    After computing the squared distance between the inputs, the mean value over
    the last dimension is returned.

    ```python
    loss = mean(square(target - preds), axis=-1)
    ```

    Usage:

    ```python
    rng = jax.random.PRNGKey(42)

    target = jax.random.randint(rng, shape=(2, 3), minval=0, maxval=2)
    preds = jax.random.uniform(rng, shape=(2, 3))

    loss = tx.losses.mean_squared_error(target, preds)

    assert loss.shape == (2,)

    assert jnp.array_equal(loss, jnp.mean(jnp.square(target - preds), axis=-1))
    ```

    Arguments:
        target: Ground truth values. shape = `[batch_size, d0, .. dN]`.
        preds: The predicted values. shape = `[batch_size, d0, .. dN]`.

    Returns:
        Mean squared error values. shape = `[batch_size, d0, .. dN-1]`.
    """

    target = target.astype(preds.dtype)

    return jnp.mean(jnp.square(preds - target), axis=-1)


# Keeping but not refactored to flax/optax yet
def line_search (direction, loss_fn, model, batches, seed, ub):
    direction /= np.linalg.norm(direction)

    def nested_loss_wrap ():
        this_weights, unravel = fu.ravel_pytree(model)
        inputs, targets = batches

        def loss_wrap (p):
            weights = this_weights + direction * p
            this_model = unravel(weights)
            preds = this_model(inputs)
            return jax.numpy.sum(loss_fn.call(targets, preds)) / len(targets)

        return loss_wrap

    loss_wrap = nested_loss_wrap()
    p_opt = glob_opt(loss_wrap, [0], method="Nelder-Mead", bounds=[(-ub, ub)])
    return p_opt['x'], p_opt['fun']


class Lambda(nn.Module):
    """
    A Module that applies a pure function to its input.
    """

    f: tp.Callable[[jnp.ndarray], jnp.ndarray]

    def __call__ (self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Arguments:
            x: The input to the function.
        Returns:
            The output of the function.
        """
        return self.f(x)


import flax.linen as nn
import jax.numpy as jnp
from functools import partial


class ResBlock(nn.Module):
    out_channels: int
    strides: int = 1
    depth: int = 2
    p_drop: float = 0.1
    dropout: bool = True
    bn: bool = True
    sc_conv: str = "Identity"
    default_kernel_init: tp.Callable = nn.initializers.lecun_normal()

    @nn.compact
    def __call__ (self, x, train: bool = True):
        # Shortcut connection
        sc = x
        if self.strides > 1:
            if "Iden" in self.sc_conv:
                # Identity shortcut with strided subsampling
                sc = jnp.pad(
                    x[:, ::self.strides, ::self.strides, :],
                    pad_width=((0, 0), (0, 0), (0, 0),
                               ((self.out_channels - x.shape[-1]) // 2,
                                self.out_channels - x.shape[-1] - (self.out_channels - x.shape[-1]) // 2)),
                    mode="constant"
                )
            elif "Conv" in self.sc_conv:
                sc = nn.Conv(self.out_channels, kernel_size=(1, 1), strides=self.strides, use_bias=False,
                             kernel_init=self.default_kernel_init)(sc)
                if self.bn:
                    sc = nn.BatchNorm(use_running_average=not train)(sc)
            elif "Line" in self.sc_conv:
                sc = sc[:, ::self.strides, ::self.strides, :]
                sc = nn.Dense(self.out_channels)(sc)

        # Main block layers
        for i in range(self.depth):
            x = nn.Conv(self.out_channels, kernel_size=(3, 3), strides=(self.strides if i == 0 else 1),
                        use_bias=False, kernel_init=self.default_kernel_init)(x)
            if self.bn:
                x = nn.BatchNorm(use_running_average=not train)(x)
            if self.dropout:
                x = nn.Dropout(self.p_drop)(x, deterministic=not train)
            x = nn.relu(x)  # Activation

        # Residual connection
        x = nn.relu(x + sc)
        return x


class ResNet(nn.Module):
    """A ResNet model."""
    n_blocks: int = 3
    layers_per_block: int = 3
    depth_per_layer: int = 2
    base_width: int = 8
    width_expansion_factor: int = 2

    use_DO: bool = False
    use_BN: bool = True
    n_out: int = 10
    p_drop: float = 0.1
    sc_conv: tp.Any = 'Identity'
    deterministic: tp.Optional[bool] = None

    @nn.compact
    def __call__ (self, x, train=True):
        x = nn.Conv(self.base_width, [3, 3], strides=[1, 1], use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x) if self.use_BN else Lambda(f=lambda x: x)(x)
        x = nn.Dropout(0.1)(x, deterministic=not train) if self.use_DO else Lambda(f=lambda x: x)(x)
        x = Lambda(jax.nn.relu)(x)

        for i in range(self.n_blocks):
            width_factor = self.width_expansion_factor ** i
            for j in range(self.layers_per_block):
                strides = 2 if (i > 0 and j == 0) else 1
                x = ResBlock(out_channels=width_factor * self.base_width, strides=strides, depth=self.depth_per_layer,
                             dropout=self.use_DO, bn=self.use_BN, p_drop=self.p_drop, sc_conv=self.sc_conv)(x,
                                                                                                            train)  # 3

        x = partial(jnp.mean, axis=(1, 2))(x)
        x = nn.Dense(self.n_out)(x)

        return x


def get_sgd_optimizer(
    learning_rate: ScalarOrSchedule,
    b1: float,
    b2: float,
    b3: Optional[float] = None,
    weight_decay: float = 0.0,
    norm_clip=None,
    verbose: bool = False,
    debug_one: bool = False,  # Not used yet, but kept for compatibility
) -> base.GradientTransformation:
    optims = []

    if norm_clip is not None:
        optims.append(optax.clip_by_global_norm(norm_clip))

    if b1 == 0 and b2 == 0:
        if verbose:
            print("Using SGD")
        if weight_decay > 0:
            optims.append(optax.add_decayed_weights(weight_decay))
        optims.append(optax.sgd(learning_rate))

    elif b1 != 0 and b2 == 0:
        if verbose:
            print("Using Adam (momentum only)")
        if weight_decay > 0:
            optims.append(optax.adamw(learning_rate, b1=b1, b2=0, eps=0., eps_root=1., weight_decay=weight_decay))
        else:
            optims.append(optax.adam(learning_rate, b1=b1, b2=0, eps=0., eps_root=1.))

    elif b2 != 0:
        # Determine custom Adam mode from b3
        if b3 != 0.:
            if b3 > 0:
                mode = ">="
            elif b3 < 0:
                mode = "<="
            if verbose:
                print(f"Using custom bounded Adam mode: {mode}")
            optims.append(oab.adam_oab(
                mode=mode,
                learning_rate=learning_rate,
                b1=b1,
                b2=b2,
                eps=1e-8,
                eps_root=0.0,
                weight_decay=weight_decay,
                nesterov=False
            ))
        else:
            if verbose:
                print("Using standard Adam")
            if weight_decay > 0:
                optims.append(optax.adamw(learning_rate, b1=b1, b2=b2, weight_decay=weight_decay))
            else:
                optims.append(optax.adam(learning_rate, b1=b1, b2=b2))

    return optax.chain(*optims)

from typing import Any, Callable

from flax import core
from flax import struct
import optax


class TrainStateSAM(struct.PyTreeNode):
    '''
    Taken from flax, adpated for SAM
    '''
    step: int
    apply_fn: Callable = struct.field(pytree_node=False)
    params: core.FrozenDict[str, Any] = struct.field(pytree_node=True)
    tx: optax.GradientTransformation = struct.field(pytree_node=False)
    opt_state: optax.OptState = struct.field(pytree_node=True)

    def apply_gradients_SAM (self, *, grads, loss_wrap, **kwargs):
        grad_fn = jax.grad(lambda p, _: loss_wrap(p))
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params, grad_fn=grad_fn)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    def apply_gradients (self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        )

    @classmethod
    def create (cls, *, apply_fn, params, tx, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )


class VGGBlock(nn.Module):
    out_channels: int
    strides: int = 1
    depth: int = 2
    p_drop: float = 0.1
    dropout: bool = True
    kernel: tuple = (3, 3)
    bn: bool = True
    pool: str = "max"
    default_kernel_init: tp.Callable = nn.initializers.lecun_normal()

    @nn.compact
    def __call__ (self, x, train: bool = True):
        for i in range(self.depth):
            x = nn.Conv(self.out_channels, self.kernel, strides=self.strides, kernel_init=self.default_kernel_init)(x)
            if self.bn:
                x = nn.BatchNorm(use_running_average=not train)(x)
            if self.dropout:
                x = nn.Dropout(self.p_drop)(x, deterministic=not train)
            x = nn.relu(x)  # Activation
        if self.pool == "max":
            x = max_pool(x, (2, 2), strides=(2, 2))
        elif self.pool == "ave":
            x = ave_pool(x, (2, 2), strides=(2, 2))
        else:
            raise NotImplementedError

        return x


class VGGNet(nn.Module):
    """A VGG model."""
    n_blocks: int = 3
    layers_per_block: int = 3
    base_width: int = 8
    width_expansion_factor: int = 2
    n_out: int = 10
    pool: str = 'max'

    use_DO: bool = False
    use_BN: bool = True
    deterministic: tp.Optional[bool] = None

    @nn.compact
    def __call__ (self, x, train=True):

        for i in range(self.n_blocks):
            width_factor = self.width_expansion_factor ** i
            x = VGGBlock(out_channels=width_factor * self.base_width, depth=self.layers_per_block,
                         pool=self.pool, dropout=self.use_DO, bn=self.use_BN)(x, train)

        x = x.reshape(x.shape[0], -1)  # flatten
        x = nn.Dense(width_factor)(x)
        if self.use_BN:
            x = nn.BatchNorm(use_running_average=not train)(x)
        if self.use_DO:
            x = nn.Dropout(self.p_drop)(x, deterministic=not train)
        x = nn.relu(x)
        x = nn.Dense(self.n_out)(x)

        return x

class MLPNet(nn.Module):
    """An MLP model."""
    depth: int = 3
    n_h: int = 32
    n_out: int = 10

    use_DO: bool = False
    use_BN: bool = True
    inputs_flatten: bool = True
    deterministic: tp.Optional[bool] = None

    @nn.compact
    def __call__ (self, x, train=True):
        if self.inputs_flatten:
            x = x.reshape((x.shape[0], -1))

        for i in range(self.depth):
            x = nn.Dense(self.n_h)(x)
            if self.use_BN:
                x = nn.BatchNorm(use_running_average=not train)(x)
            if self.use_DO:
                x = nn.Dropout(self.p_drop)(x, deterministic=not train)
            x = nn.relu(x)

        x = nn.Dense(self.n_out)(x)

        return x


import optax

def build_lr_schedule(cfg, is_adv=False):
    """
    Creates a warmup + decay learning rate scheduler.
    """
    lr_mode = cfg.optim.lr_decay_mode
    peak_lr = cfg.optim.sam_rho * cfg.optim.lr if is_adv else cfg.optim.lr
    warmup_steps = cfg.optim.warmup_steps
    total_steps = cfg.optim.n_epochs
    decay_steps = total_steps - warmup_steps

    warmup = optax.linear_schedule(
        init_value=0.0,
        end_value=peak_lr,
        transition_steps=warmup_steps
    )

    if lr_mode == 'constant':
        decay = optax.constant_schedule(peak_lr)
    elif lr_mode == 'cosine':
        decay = optax.cosine_decay_schedule(
            init_value=peak_lr,
            decay_steps=decay_steps,
            alpha=0.1  # final LR = 0
        )
    elif lr_mode == 'linear':
        decay = optax.linear_schedule(
            init_value=peak_lr,
            end_value=peak_lr*0.1,
            transition_steps=decay_steps
        )
    else:
        raise NotImplementedError(f"Unknown LR decay mode: {lr_mode}")

    return optax.join_schedules([warmup, decay], boundaries=[warmup_steps])
