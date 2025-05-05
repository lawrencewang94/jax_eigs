import functools
import typing
from typing import Any, Optional, Union, Callable, NamedTuple, Protocol

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import flatten_util as fu
from jax import tree_util as tu

import optax
from optax._src import base, combine, numerics, transform

# Type aliases
DType = np.dtype
ScalarOrSchedule = Union[float, base.Schedule]


@typing.runtime_checkable
class SupportsDType(Protocol):
    @property
    def dtype(self) -> DType: ...


DTypeLike = Union[
    str,
    type[Any],
    np.dtype,
    SupportsDType,
]


def _canonicalize_dtype(dtype: Optional[chex.ArrayDType]) -> Optional[chex.ArrayDType]:
    return jax.dtypes.canonicalize_dtype(dtype) if dtype is not None else None


def _tree_ones_like(tree: Any, dtype: Optional[DTypeLike] = None) -> Any:
    return jax.tree_map(lambda x: jnp.ones_like(x, dtype=dtype), tree)


def _tree_zeros_like(tree: Any, dtype: Optional[DTypeLike] = None) -> Any:
    return jax.tree_map(lambda x: jnp.zeros_like(x, dtype=dtype), tree)


def _tree_update_moment(updates, moments, decay, order):
    return jax.tree_map(
        lambda g, t: (1 - decay) * (g ** order) + decay * t if g is not None else None,
        updates,
        moments,
        is_leaf=lambda x: x is None,
    )


def _tree_update_moment_per_elem_norm(updates, moments, decay, order):
    def orderth_norm(g):
        if jnp.isrealobj(g):
            return g ** order
        half_order = order / 2
        if half_order.is_integer():
            half_order = int(half_order)
        return numerics.abs_sq(g) ** half_order

    return jax.tree_map(
        lambda g, t: (1 - decay) * orderth_norm(g) + decay * t if g is not None else None,
        updates,
        moments,
        is_leaf=lambda x: x is None,
    )


def _safe_increment(count: chex.Numeric) -> chex.Numeric:
    count_dtype = jnp.asarray(count).dtype
    if jnp.issubdtype(count_dtype, jnp.integer):
        max_value = jnp.iinfo(count_dtype).max
    elif jnp.issubdtype(count_dtype, jnp.floating):
        max_value = jnp.finfo(count_dtype).max
    else:
        raise ValueError(f"Unsupported dtype {count_dtype} for count increment.")
    return jnp.minimum(count + 1, max_value)


@functools.partial(jax.jit, inline=True)
def _tree_bias_correction(moment, decay, count):
    bias_correction = 1 - decay ** count
    return jax.tree_map(lambda t: t / bias_correction.astype(t.dtype), moment)


def _tree_cast(tree: chex.ArrayTree, dtype: Optional[chex.ArrayDType]) -> chex.ArrayTree:
    return jax.tree_map(lambda t: t.astype(dtype), tree) if dtype is not None else tree


def _scale_by_learning_rate(learning_rate: ScalarOrSchedule, flip_sign=True):
    sign = -1 if flip_sign else 1
    if callable(learning_rate):
        return transform.scale_by_schedule(lambda count: sign * learning_rate(count))
    return transform.scale(sign * learning_rate)


def tree_mult_by_scalar(pytree, scalar):
    return jax.tree_map(lambda x: x * scalar, pytree)


class ScaleByAdamState(NamedTuple):
    count: chex.Array
    mu: base.Updates
    nu: base.Updates


def _scale_by_adam_bound(
    b1: float,
    b2: float,
    eps: float,
    eps_root: float,
    bound_fn: Callable[[Any, Any, float], Any],
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformation:
    mu_dtype = _canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = _tree_zeros_like(params, dtype=mu_dtype)
        nu = _tree_zeros_like(params)
        return ScaleByAdamState(count=jnp.zeros([], jnp.int32), mu=mu, nu=nu)

    def update_fn(updates, state, params=None):
        del params
        mu = _tree_update_moment(updates, state.mu, b1, 1)
        nu = _tree_update_moment_per_elem_norm(updates, state.nu, b2, 2)
        count_inc = _safe_increment(state.count)

        if nesterov:
            mu_hat = jax.tree_map(
                lambda m, g: b1 * m + (1 - b1) * g,
                _tree_bias_correction(mu, b1, _safe_increment(count_inc)),
                _tree_bias_correction(updates, b1, count_inc),
            )
        else:
            mu_hat = _tree_bias_correction(mu, b1, count_inc)

        nu_hat = _tree_bias_correction(nu, b2, count_inc)

        updates = jax.tree_map(
            lambda m, v: None if m is None else m / (bound_fn(v, eps_root) + eps),
            mu_hat, nu_hat, is_leaf=lambda x: x is None,
        )
        mu = _tree_cast(mu, mu_dtype)
        return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)


def scale_by_adam_const(**kwargs):
    return _scale_by_adam_bound(
        bound_fn=lambda v, eps_root: tree_mult_by_scalar(_tree_ones_like(v), 1.0),
        **kwargs
    )


def scale_by_adam_clip_below(lower_bound: float = 1.0, **kwargs):
    return _scale_by_adam_bound(
        bound_fn=lambda v, eps_root: jax.tree_map(jnp.maximum, jnp.sqrt(v + eps_root), tree_mult_by_scalar(_tree_ones_like(v), lower_bound)),
        **kwargs
    )


def scale_by_adam_clip_above(upper_bound: float = 1.0, **kwargs):
    return _scale_by_adam_bound(
        bound_fn=lambda v, eps_root: jax.tree_map(jnp.minimum, jnp.sqrt(v + eps_root), tree_mult_by_scalar(_tree_ones_like(v), upper_bound)),
        **kwargs
    )


def adam_oab(
    mode: str,
    learning_rate: ScalarOrSchedule,
    b1: float = 0.9,
    b2: float = 0.99,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    weight_decay: float = 0.0,
    mu_dtype: Optional[Any] = None,
    mask: Optional[Union[Any, Callable[[base.Params], Any]]] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformation:
    mode_to_transform = {
        "==": scale_by_adam_const,
        ">=": scale_by_adam_clip_below,
        "<=": scale_by_adam_clip_above,
    }

    if mode not in mode_to_transform:
        raise ValueError(f"Invalid mode: {mode}. Choose from {list(mode_to_transform)}")

    transforms = []

    transforms.append(mode_to_transform[mode](
            b1=b1, b2=b2, eps=eps, eps_root=eps_root,
            mu_dtype=mu_dtype, nesterov=nesterov
        ))

    if weight_decay > 0:
        transforms.append(optax.add_decayed_weights(weight_decay, mask))

    transforms.append(_scale_by_learning_rate(learning_rate))

    return combine.chain(*transforms)
