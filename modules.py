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

# python file for loss functions, optimizers, computations, etc.


# @jax.jit
def lognormpdf(x, mean, std):
    log = jnp.log
    # assumes diagonal covariance so 1D mean and 1D cov vectors
    # bounding the std - see deepmind notebook https://github.com/deepmind/neural-processes/blob/master/conditional_neural_process.ipynb
    std = 0.01 + 0.99 * std
    variance = std ** 2
    variation = x - mean
    out = -log(std) - 0.5 * log(2 * pi) - variation ** 2 / (2 * variance)
    return out


def gaussNLL(
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
def ntk2hess(ntk, norm=True):
    k = ntk.shape[2]
    B = ntk.shape[0]
    assert ntk.shape[0] == ntk.shape[1]
    assert ntk.shape[2] == ntk.shape[3]
    hess = np.transpose(ntk, axes=(0, 2, 1, 3))
    hess = np.reshape(hess, (ntk.shape[0] * ntk.shape[2], ntk.shape[0] * ntk.shape[2]))
    if norm:
        hess /= (k*B)
    return hess


# Keeping but not refactored to flax/optax yet
def compute_empirical_NTK(model, tr_inputs, method=1, hessian=False):

    def model_call(model):
        _, ravel = fu.ravel_pytree(model)
        def call(weights, inputs):
            model_new = ravel(weights)
            return model_new(inputs)
        return call

    kwargs = dict(
      f=model_call(model),
      trace_axes=(),
      vmap_axes=0
    )

    ntk_fn = jax.jit(nt.empirical_ntk_fn(** kwargs, implementation=method))
    x1 = np.array(tr_inputs)

    params, _ = fu.ravel_pytree(model)
    ntk = ntk_fn(x1, None, params)
    hess = ntk2hess(ntk)
    w_ntk, v_ntk = np.linalg.eig(hess)
    if hessian:
        hessian =  jax.grad
    # print(w_ntk)
    # print("2/lam0", 2/np.max(w_ntk), "4/lam0", 4/np.max(w_ntk), "12/lam0", 12/np.max(w_ntk))
    return w_ntk


def pool(inputs, init, reduce_fn, window_shape, strides, padding):
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


def max_pool(inputs, window_shape, strides=None, padding="VALID"):
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


def ave_pool(inputs, window_shape, strides=None, padding="VALID"):
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


def mean_squared_error(target: jnp.ndarray, preds: jnp.ndarray) -> jnp.ndarray:
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
def line_search(direction, loss_fn, model, batches, seed, ub):
    direction /= np.linalg.norm(direction)

    def nested_loss_wrap():
        this_weights, unravel = fu.ravel_pytree(model)
        inputs, targets = batches

        def loss_wrap(p):
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

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Arguments:
            x: The input to the function.
        Returns:
            The output of the function.
        """
        return self.f(x)


import flax.linen as nn
import jax.numpy as jnp

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
    def __call__(self, x, train: bool = True):
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


# class ResBlock(nn.Module):
#
#     out_channels: int
#
#     # layers: tp.List[CallableModule] = dataclasses.field(default_factory=list)
#     # shortcut_layers: tp.List[CallableModule] = dataclasses.field(default_factory=list)
#
#     strides: int = 1
#     w_expand: int = 2 # width expansion ratio
#     depth: int = 2
#     p_drop: float = 0.1
#     dropout: bool = True
#     bn: bool = True
#     sc_conv: str = 'Identity'
#     default_kernel_init = lecun_normal
#     deterministic: tp.Optional[bool] = None
#
#     def setup(self,):
#         """
#         Arguments:
#             *layers: A list of layers or callables to apply in sequence.
#         """
#         # legacy: True = Conv, False = Identity
#         # assert self.depth > 0
#         # assert self.sc_conv is not None
#         #
#         # self.layers: tp.List[CallableModule] = dataclasses.field(default_factory=list)
#         # self.shortcut_layers: tp.List[CallableModule] = dataclasses.field(default_factory=list)
#         if isinstance(self.sc_conv, (int)):
#             if self.sc_conv:
#                 this_sc_conv = 'Conv'
#             else:
#                 this_sc_conv = 'Identity'
#         else:
#             this_sc_conv = self.sc_conv
#
#         layers = [
#             nn.Conv(self.out_channels, [3, 3], strides=[self.strides, self.strides], use_bias=False, kernel_init=self.default_kernel_init),
#             nn.BatchNorm() if self.bn else lambda x:x,
#             nn.Dropout(self.p_drop) if self.dropout else lambda x:x,
#         ]
#
#         for i in range(self.depth-1):
#             layers.extend([
#                 jax.nn.relu,
#                 nn.Conv(self.out_channels, [3, 3], strides=[self.strides, self.strides], use_bias=False, kernel_init=self.default_kernel_init),
#                 nn.BatchNorm() if self.bn else lambda x: x,
#                 nn.Dropout(self.p_drop) if self.dropout else lambda x: x,
#             ])
#         layers = [
#             layer if isinstance(layer, nn.Module) else Lambda(f=layer) for layer in layers
#         ]
#         self.layers = layers
#
#         if self.strides > 1:
#             if this_sc_conv[:4] == 'Iden':
#                 # subsample the image by strides, and pad the feature vectors
#                 # in_channels = x.shape[-1]
#                 # n_pad0 = (out_channels-in_channels)/2, n_pad1 = out_channels - n_pad_0 - in_channels
#                 self.shortcut_layers = [
#                     lambda x: jnp.pad(x[:, ::self.strides, ::self.strides, :], ((0, 0), (0, 0), (0, 0),
#                             ((self.out_channels - x.shape[-1]) // 2, self.out_channels - x.shape[-1] - (self.out_channels - x.shape[-1]) // 2)),
#                             mode='constant'),
#                 ]
#             elif this_sc_conv[:4] == 'Conv':
#                 self.shortcut_layers = [
#                     nn.Conv(self.out_channels, [1, 1], strides=self.strides, use_bias=False, kernel_init=self.default_kernel_init),
#                     nn.BatchNorm() if self.bn else lambda x:x,
#                 ]
#             elif this_sc_conv[:4] == 'Line':
#                 # subsample the image
#                 # use same linear network to share filters
#                 self.shortcut_layers = [
#                     lambda x: x[:, ::self.strides, ::self.strides, :],
#                     nn.Dense(self.out_channels),
#                 ]
#         else:
#             self.shortcut_layers = [
#                 lambda x: x,
#                 ]
#         self.shortcut_layers = [
#             layer if isinstance(layer, nn.Module) else Lambda(layer) for layer in self.shortcut_layers
#         ]
#
#     @nn.compact
#     def __call__(self, x, deterministic=None):
#         deterministic = nn.merge_param('deterministic', self.deterministic, deterministic)
#         sc = x
#         for sl in self.shortcut_layers:
#             sc = sl(x)
#         for layer in self.layers:
#             x = layer(x)
#         x = jax.nn.relu(x + sc)
#         return x

#
# class VGGBlock(nn.Module):
#     layers: tp.List[CallableModule] = to.node()
#
#     def __init__(
#             self, out_channels, depth=2, strides=1, p_drop=0.1, dropout=True, bn=True
#     ):
#         """
#         Arguments:
#             *layers: A list of layers or callables to apply in sequence.
#         """
#         self.out_channels = out_channels
#         self.strides = strides
#         self.depth = depth
#         assert depth > 0
#
#         self.layers = [
#             tx.Conv(out_channels, [3, 3], strides=[self.strides, self.strides]),
#             tx.BatchNorm() if bn else lambda x:x,
#             tx.Dropout(p_drop) if dropout else lambda x:x,
#             jax.nn.relu,
#         ]
#
#         for i in range(depth-1):
#             self.layers += [
#                 tx.Conv(out_channels, [3, 3], strides=[1, 1]),
#                 tx.BatchNorm() if bn else lambda x: x,
#                 tx.Dropout(p_drop) if dropout else lambda x: x,
#                 jax.nn.relu,
#             ]
#
#         self.layers += [
#             lambda x: max_pool(x, (2, 2), strides=(2, 2)),
#         ]
#
#         self.layers = [
#             layer if isinstance(layer, Module) else Lambda(layer) for layer in self.layers
#         ]
#
#     def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
#         for layer in self.layers:
#             x = layer(x)
#
#         return x
#
#
# class AlexNet(nn.Module):
#     layers: tp.List[CallableModule] = to.node()
#
#     def __init__(
#             self, conv0, conv1, fc0, fc1, p_drop=0.1, dropout=True, bn=True
#     ):
#         """
#         Arguments:
#             *layers: A list of layers or callables to apply to apply in sequence.
#         """
#         self.layers = [
#             tx.Conv(conv0, [5, 5], strides=[2, 2]),
#             tx.BatchNorm() if bn else lambda x:x,
#             tx.Dropout(p_drop) if dropout else lambda x:x,
#             jax.nn.relu,
#             lambda x: max_pool(x, (3, 3)),
#
#             tx.Conv(conv1, [5, 5], strides=[2, 2]),
#             tx.BatchNorm() if bn else lambda x:x,
#             tx.Dropout(p_drop) if dropout else lambda x:x,
#             jax.nn.relu,
#             lambda x: max_pool(x, (3, 3)),
#
#             lambda x: x.reshape(x.shape[0], -1), # flatten
#             tx.Linear(fc0),
#             tx.BatchNorm() if bn else lambda x:x,
#             tx.Dropout(p_drop) if dropout else lambda x:x,
#             jax.nn.relu,
#
#             tx.Linear(fc1),
#             tx.BatchNorm() if bn else lambda x:x,
#             tx.Dropout(p_drop) if dropout else lambda x:x,
#             jax.nn.relu,
#
#         ]
#         # self.layers = [
#         #     layer if isinstance(layer, Module) else Lambda(layer) for layer in self.layers
#         # ]
#
#     def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
#         for layer in self.layers:
#             x = layer(x)
#
#         return x

# Keeping but not refactored to flax/optax yet
# class TransformerCE():
#     def __init__(self,
#                  *,
#                  from_logits: bool = True,
#                  binary: bool = False,
#                  label_smoothing: tp.Optional[float] = None,
#                  reduction: tp.Optional[Reduction] = None,
#                  check_bounds: bool = True,
#                  weight: tp.Optional[float] = None,
#                  on: tp.Optional[types.IndexLike] = None,
#                  name: tp.Optional[str] = None,
#                  ):
#         super().__init__(reduction=reduction, weight=weight, on=on, name=name)
#         self.loss_fn = tx.losses.Crossentropy()
#
#     def call(self, target: jnp.ndarray, preds: jnp.ndarray, sample_weight: tp.Optional[jnp.ndarray] = None, ) -> jnp.ndarray:
#         B, T, C = preds.shape
#         return self.loss_fn(preds=preds.reshape(B*T, C), target=target.reshape(B*T, 1), sample_weight=sample_weight)

#
# # Keeping but not refactored to flax/optax yet
# class MeanSquaredErrorOneHot():
#     """
#     Computes the mean of squares of errors between target and predictions.
#
#     `loss = square(target - preds)`
#
#     Usage:
#
#     ```python
#     target = jnp.array([[0.0, 1.0], [0.0, 0.0]])
#     preds = jnp.array([[1.0, 1.0], [1.0, 0.0]])
#
#     # Using 'auto'/'sum_over_batch_size' reduction type.
#     mse = tx.losses.MeanSquaredError()
#
#     assert mse(target, preds) == 0.5
#
#     # Calling with 'sample_weight'.
#     assert mse(target, preds, sample_weight=jnp.array([0.7, 0.3])) == 0.25
#
#     # Using 'sum' reduction type.
#     mse = tx.losses.MeanSquaredError(reduction=tx.losses.Reduction.SUM)
#
#     assert mse(target, preds) == 1.0
#
#     # Using 'none' reduction type.
#     mse = tx.losses.MeanSquaredError(reduction=tx.losses.Reduction.NONE)
#
#     assert list(mse(target, preds)) == [0.5, 0.5]
#     ```
#     Usage with the Elegy API:
#
#     ```python
#     model = elegy.Model(
#         module_fn,
#         loss=tx.losses.MeanSquaredError(),
#         metrics=elegy.metrics.Mean(),
#     )
#     ```
#     """
#
#     def __init__(
#         self,
#         n_classes: jnp.int32 = 10,
#         reduction: tp.Optional[Reduction] = None,
#         weight: tp.Optional[float] = None,
#         on: tp.Optional[types.IndexLike] = None,
#         name: tp.Optional[str] = None,
#     ):
#         """
#         Initializes `Mean` class.
#
#         Arguments:
#             reduction: (Optional) Type of `tx.losses.Reduction` to apply to
#                 loss. Default value is `SUM_OVER_BATCH_SIZE`. For almost all cases
#                 this defaults to `SUM_OVER_BATCH_SIZE`.
#             weight: Optional weight contribution for the total loss. Defaults to `1`.
#             on: A string or integer, or iterable of string or integers, that
#                 indicate how to index/filter the `target` and `preds`
#                 arguments before passing them to `call`. For example if `on = "a"` then
#                 `target = target["a"]`. If `on` is an iterable
#                 the structures will be indexed iteratively, for example if `on = ["a", 0, "b"]`
#                 then `target = target["a"][0]["b"]`, same for `preds`. For more information
#                 check out [Keras-like behavior](https://poets-ai.github.io/elegy/guides/modules-losses-metrics/#keras-like-behavior).
#         """
#         self.n_classes = n_classes
#         return super().__init__(reduction=reduction, weight=weight, on=on, name=name)
#
#     def call(
#         self,
#         target: jnp.ndarray,
#         preds: jnp.ndarray,
#         sample_weight: tp.Optional[
#             jnp.ndarray
#         ] = None,  # not used, __call__ handles it, left for documentation purposes.
#     ) -> jnp.ndarray:
#         """
#         Invokes the `MeanSquaredError` instance.
#
#         Arguments:
#             target: Ground truth values. shape = `[batch_size, d0, .. dN]`, except
#                 sparse loss functions such as sparse categorical crossentropy where
#                 shape = `[batch_size, d0, .. dN-1]`
#             preds: The predicted values. shape = `[batch_size, d0, .. dN]`
#             sample_weight: Optional `sample_weight` acts as a
#                 coefficient for the loss. If a scalar is provided, then the loss is
#                 simply scaled by the given value. If `sample_weight` is a tensor of size
#                 `[batch_size]`, then the total loss for each sample of the batch is
#                 rescaled by the corresponding element in the `sample_weight` vector. If
#                 the shape of `sample_weight` is `[batch_size, d0, .. dN-1]` (or can be
#                 broadcasted to this shape), then each loss element of `preds` is scaled
#                 by the corresponding value of `sample_weight`. (Note on`dN-1`: all loss
#                 functions reduce by 1 dimension, usually axis=-1.)
#
#         Returns:
#             Weighted loss float `Tensor`. If `reduction` is `NONE`, this has
#                 shape `[batch_size, d0, .. dN-1]`; otherwise, it is scalar. (Note `dN-1`
#                 because all loss functions reduce by 1 dimension, usually axis=-1.)
#
#         Raises:
#             ValueError: If the shape of `sample_weight` is invalid.
#         """
#         return mean_squared_error(jax.nn.one_hot(target, self.n_classes), preds) ## CHANGED LINE


def get_sgd_optimizer(lr_schedule, b1, b2, b3=None, verbose=False, debug_one=False):
    if b1 == 0 and b2 == 0:
        if verbose: print("Using vanilla SGD!")
        optimizer = optax.sgd(lr_schedule)
    elif b1 != 0 and b2 == 0:
        # if verbose: print("Using SGD momentum!")
        # optimizer = tx.Optimizer(optax.sgd(lr_scheduler, momentum=b1))
        ### Now we use adam for EMA scaled momentum, and better comparison
        if verbose: print("Using Adam!")
        optimizer = optax.adam(lr_schedule, b1=b1, b2=0, eps_root=1., eps=0.) # eps root = 1 so we don't divide by 0 because b2 is 0
    else:
        if b3 != 0:
            import optax_adam_bounded as oab
            import importlib
            importlib.reload(oab)

        if verbose: print("Using Adam!")
        if b3 != 0 and debug_one:
            optimizer = oab.adam_one(lr_schedule, b1=b1, b2=b2)
        elif b3 > 0:
            optimizer = oab.adam_lb(lr_schedule, b1=b1, b2=b2, lower_bound=b3)
        elif b3 < 0:
            optimizer = oab.adam_ub(lr_schedule, b1=b1, b2=b2, upper_bound=abs(b3))
        else:
            optimizer = optax.adam(lr_schedule, b1=b1, b2=b2)

    return optimizer




from typing import Any, Callable

from flax import core
from flax import struct
import optax
from optax import contrib

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
        updates, new_opt_state = self.tx.update(
            grads, self.opt_state, self.params, grad_fn=jax.grad(lambda p, _: loss_wrap(p)))
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
