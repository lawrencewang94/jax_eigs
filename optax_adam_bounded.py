from optax._src import base
from optax._src import transform
import jax
import jax.numpy as jnp
from typing import Any, Optional, Union, NamedTuple, Protocol
import chex
from jax import flatten_util as fu
from jax import tree_util as tu
from optax._src import combine
from optax._src import numerics
import numpy as np
import typing
import functools

DType = np.dtype

ScalarOrSchedule = Union[float, base.Schedule]

@typing.runtime_checkable
class SupportsDType(Protocol):
    @property
    def dtype(self) -> DType: ...


DTypeLike = Union[
    str,            # like 'float32', 'int32'
    type[Any],      # like np.float32, np.int32, float, int
    np.dtype,       # like np.dtype('float32'), np.dtype('int32')
    SupportsDType,  # like jnp.float32, jnp.int32
]

def _canonicalize_dtype(
    dtype: Optional[chex.ArrayDType],
) -> Optional[chex.ArrayDType]:
    """Canonicalise a dtype, skip if None."""
    if dtype is not None:
        return jax.dtypes.canonicalize_dtype(dtype)
    return dtype


# @jax.jit
def _tree_add(tree_left, tree_right):
    """Computes tree_left + tree_right."""
    def add(x, y):
        return x + y
    # return tu.tree_multimap(add, tree_left, tree_right)

    return tu.tree_map(add, tree_left, tree_right)

def _scale_by_learning_rate(learning_rate: ScalarOrSchedule, flip_sign=True):
    m = -1 if flip_sign else 1
    if callable(learning_rate):
        return transform.scale_by_schedule(lambda count: m * learning_rate(count))
    return transform.scale(m * learning_rate)


def _tree_ones_like(
    tree: Any,
    dtype: Optional[DTypeLike] = None,
) -> Any:
    """Creates an all-ones tree with the same structure.

    Args:
    tree: pytree.
    dtype: optional dtype to use for the tree of ones.

    Returns:
    an all-ones tree with the same structure as ``tree``.
    """
    return jax.tree_map(lambda x: jnp.ones_like(x, dtype=dtype), tree)

def _tree_zeros_like(
    tree: Any,
    dtype: Optional[DTypeLike] = None,
) -> Any:
    """Creates an all-ones tree with the same structure.

    Args:
    tree: pytree.
    dtype: optional dtype to use for the tree of ones.

    Returns:
    an all-ones tree with the same structure as ``tree``.
    """
    return jax.tree_map(lambda x: jnp.zeros_like(x, dtype=dtype), tree)



def _tree_update_moment(updates, moments, decay, order):
  """Compute the exponential moving average of the `order`-th moment."""
  return jax.tree_map(
      lambda g, t: (
          (1 - decay) * (g**order) + decay * t if g is not None else None
      ),
      updates,
      moments,
      is_leaf=lambda x: x is None,
  )


def _tree_update_moment_per_elem_norm(updates, moments, decay, order):
  """Compute the EMA of the `order`-th moment of the element-wise norm."""

  def orderth_norm(g):
    if jnp.isrealobj(g):
        return g ** order

    half_order = order / 2
    # JAX generates different HLO for int and float `order`
    if half_order.is_integer():
        half_order = int(half_order)
    return numerics.abs_sq(g) ** half_order

  return jax.tree_map(
      lambda g, t: (
          (1 - decay) * orderth_norm(g) + decay * t if g is not None else None
      ),
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
        raise ValueError(
            f'Cannot safely increment count with dtype {count_dtype},'
            ' valid dtypes are subdtypes of "jnp.integer" or "jnp.floating".'
        )
    max_value = jnp.array(max_value, count_dtype)
    one = jnp.array(1, count_dtype)
    return jnp.where(count < max_value, count + one, max_value)


@functools.partial(jax.jit, inline=True)
def _tree_bias_correction(moment, decay, count):
    """Performs bias correction. It becomes a no-op as count goes to infinity."""
    # The conversion to the data type of the moment ensures that bfloat16 remains
    # bfloat16 in the optimizer state. This conversion has to be done after
    # `bias_correction_` is calculated as calculating `decay**count` in low
    # precision can result in it being rounded to 1 and subsequently a
    # "division by zero" error.
    bias_correction_ = 1 - decay**count

    # Perform division in the original precision.
    return jax.tree_map(lambda t: t / bias_correction_.astype(t.dtype), moment)


def _tree_cast(
    tree: chex.ArrayTree, dtype: Optional[chex.ArrayDType]
) -> chex.ArrayTree:
    """Cast tree to given dtype, skip if None.

    Args:
    tree: the tree to cast.
    dtype: the dtype to cast to, or None to skip.

    Returns:
    the tree, with leaves casted to dtype.

    Examples:
    >>> import jax.numpy as jnp
    >>> import optax
    >>> tree = {'a': {'b': jnp.array(1.0, dtype=jnp.float32)},
    ...         'c': jnp.array(2.0, dtype=jnp.float32)}
    >>> optax.tree_utils.tree_cast(tree, dtype=jnp.bfloat16)
    {'a': {'b': Array(1, dtype=bfloat16)}, 'c': Array(2, dtype=bfloat16)}
    """
    if dtype is not None:
        return jax.tree_map(lambda t: t.astype(dtype), tree)
    else:
        return tree


def adam_lb(
    learning_rate: ScalarOrSchedule, # importing directly cos my optax is outdated
    b1: float = 0.9,
    b2: float = 0.99,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    lower_bound: float = 1.0,
    mu_dtype: Optional[Any] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformation:

    return combine.chain(
        scale_by_adam_lb(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            lower_bound=lower_bound,
            mu_dtype=mu_dtype,
            nesterov=nesterov,
        ),
        # transform.scale_by_learning_rate(learning_rate)
        _scale_by_learning_rate(learning_rate) # optax outdated, importing directly
    )


def adam_ub(
    learning_rate: ScalarOrSchedule, # importing directly cos my optax is outdated
    b1: float = 0.9,
    b2: float = 0.99,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    upper_bound: float = 1.0,
    mu_dtype: Optional[Any] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformation:

    return combine.chain(
        scale_by_adam_ub(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            upper_bound=upper_bound,
            mu_dtype=mu_dtype,
            nesterov=nesterov,
        ),
        # transform.scale_by_learning_rate(learning_rate)
        _scale_by_learning_rate(learning_rate) # optax outdated, importing directly
    )

def adam_one(
    learning_rate: ScalarOrSchedule, # importing directly cos my optax is outdated
    b1: float = 0.9,
    b2: float = 0.99,
    eps: float = 0.,
    eps_root: float = 0.0,
    mu_dtype: Optional[Any] = None,
    *,
    nesterov: bool = False,
) -> base.GradientTransformation:

    return combine.chain(
        scale_by_adam_one(
            b1=b1,
            b2=b2,
            eps=eps,
            eps_root=eps_root,
            mu_dtype=mu_dtype,
            nesterov=nesterov,
        ),
        # transform.scale_by_learning_rate(learning_rate)
        _scale_by_learning_rate(learning_rate) # optax outdated, importing directly
    )

class ScaleByAdamState(NamedTuple):
  """State for the Adam algorithm."""

  count: chex.Array  # shape=(), dtype=jnp.int32.
  mu: base.Updates
  nu: base.Updates

def tree_mult_by_scalar(pytree, scalar):
    """
    Multiplies all leaves of a JAX PyTree by a scalar.

    Args:
        pytree: The input PyTree (e.g., nested dictionaries, lists, or arrays).
        scalar: The scalar value to multiply each leaf by.

    Returns:
        A new PyTree with each leaf multiplied by the scalar.
    """
    return jax.tree_map(lambda x: x * scalar, pytree)

# @jax.jit
def tracer_grad(grads, ema, eps):
    def tracer(g, ema):
        tracer_out = tu.tree_map(lambda p, q: p ** 2 / (q + eps), g, ema)
        flat_tracer_out, _ = fu.ravel_pytree(tracer_out)
        return jnp.sum(flat_tracer_out)

    return jax.grad(tracer, argnums=0, )(grads, ema)

def scale_by_adam_lb(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    lower_bound: float = 1.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = False,
    ) -> base.GradientTransformation:

    mu_dtype = _canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = _tree_zeros_like(params, dtype=mu_dtype)  # First moment
        nu = _tree_zeros_like(params)  # Second moment
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
            lambda m, v: None if m is None else m / (jax.tree_map(jnp.maximum, jnp.sqrt(v + eps_root), tree_mult_by_scalar(_tree_ones_like(v), lower_bound)) + eps),
            mu_hat,
            nu_hat,
            is_leaf=lambda x: x is None,
        )
        mu = _tree_cast(mu, mu_dtype)
        return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)

def scale_by_adam_ub(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    upper_bound: float = 1.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = False,
    ) -> base.GradientTransformation:

    mu_dtype = _canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = _tree_zeros_like(params, dtype=mu_dtype)  # First moment
        nu = _tree_zeros_like(params)  # Second moment
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
        # print(jnp.max(jnp.array(nu_hat)))
        updates = jax.tree_map(
            # lambda m, v: None if m is None else m / (jnp.min(jnp.array([jnp.sqrt(v + eps_root), tree_mult_by_scalar(_tree_ones_like(v), upper_bound)])) + eps),
            lambda m, v: None if m is None else m / (jax.tree_map(jnp.minimum, jnp.sqrt(v + eps_root), tree_mult_by_scalar(_tree_ones_like(v), upper_bound)) + eps),
            mu_hat,
            nu_hat,
            is_leaf=lambda x: x is None,
        )
        mu = _tree_cast(mu, mu_dtype)
        return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)



def scale_by_adam_one(
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    mu_dtype: Optional[chex.ArrayDType] = None,
    *,
    nesterov: bool = False,
    ) -> base.GradientTransformation:

    mu_dtype = _canonicalize_dtype(mu_dtype)

    def init_fn(params):
        mu = _tree_zeros_like(params, dtype=mu_dtype)  # First moment
        nu = _tree_zeros_like(params)  # Second moment
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
        # print(jnp.max(jnp.array(nu_hat)))
        updates = jax.tree_map(
            # lambda m, v: None if m is None else m / (jnp.min(jnp.array([jnp.sqrt(v + eps_root), tree_mult_by_scalar(_tree_ones_like(v), upper_bound)])) + eps),
            lambda m, v: None if m is None else m / (tree_mult_by_scalar(_tree_ones_like(v), 1.) + eps),
            mu_hat,
            # nu_hat,
            # mu,
            nu,
            is_leaf=lambda x: x is None,
        )
        mu = _tree_cast(mu, mu_dtype)
        return updates, ScaleByAdamState(count=count_inc, mu=mu, nu=nu)

    return base.GradientTransformation(init_fn, update_fn)

