from typing import Optional
from optax._src import base
import jax
import jax.numpy as jnp
import chex
from collections.abc import Callable
from optax._src import update
from optax.contrib import SAMState
#
# @chex.dataclass
# class SAMState:
#   """State of `GradientTransformation` returned by `sam`.
#
#   Attributes:
#     steps_since_sync: Number of adversarial steps taken since the last outer
#       update.
#     opt_state: State of the outer optimizer.
#     adv_state: State of the inner adversarial optimizer.
#     cache: a place to store the last outer updates.
#   """
#
#     steps_since_sync: jax.Array
#     opt_state: base.OptState
#     adv_state: base.OptState
#     cache: Optional[base.Params]


def sam(
        optimizer: base.GradientTransformation,
        adv_optimizer: base.GradientTransformation,
        sync_period: int = 2,
        reset_state: bool = True,
        opaque_mode: bool = False,
        batch_axis_name: Optional[str] = None,
) -> base.GradientTransformationExtraArgs:
    if sync_period < 1:
        raise ValueError ("Synchronization period must be >= 1.")

    def init_fn(params: base.Params) -> SAMState:
        return SAMState (
            steps_since_sync=jnp.zeros (shape=(), dtype=jnp.int32),
            opt_state=optimizer.init (params),
            adv_state=adv_optimizer.init (params),
            cache=None if opaque_mode else params,
        )

    def opaque_update_fn(
            updates: base.Updates,
            state: SAMState,
            params: Optional[base.Params],
            *,
            grad_fn: Optional[Callable[[base.Params, int], base.Updates]] = None,
    ) -> tuple[base.Updates, SAMState]:
        if grad_fn is None:
            raise ValueError ("grad_fn must be provided when opaque_mode=True.")

        outer_params = params
        adv_params = params
        adv_updates = updates
        adv_state = state.adv_state

        for i in range (sync_period - 1):
            adv_updates, adv_state = adv_optimizer.update (
                adv_updates, adv_state, adv_params
            )
            adv_updates = jax.tree.map (lambda x: -x, adv_updates)

            adv_params = update.apply_updates (adv_params, adv_updates)
            adv_updates = grad_fn (adv_params, i)
            if batch_axis_name is not None:
                adv_updates = jax.lax.pmean (adv_updates, axis_name=batch_axis_name)

        if reset_state:
            adv_state = adv_optimizer.init (outer_params)

        updates, opt_state = optimizer.update (
            adv_updates, state.opt_state, outer_params
        )

        return updates, SAMState (
            steps_since_sync=jnp.zeros (shape=(), dtype=jnp.int32),
            adv_state=adv_state,
            opt_state=opt_state,
            cache=None,
        )

    update_fn = opaque_update_fn

    return base.GradientTransformationExtraArgs (init_fn, update_fn)


# @chex.dataclass
# class ASAMState(SAMState):


def asam(
        optimizer: base.GradientTransformation,
        adv_optimizer: base.GradientTransformation,
        sync_period: int = 2,
        reset_state: bool = True,
        opaque_mode: bool = False,
        batch_axis_name: Optional[str] = None,
) -> base.GradientTransformationExtraArgs:
    if sync_period < 1:
        raise ValueError("Synchronization period must be >= 1.")

    def init_fn(params: base.Params) -> SAMState:
        return SAMState(
            steps_since_sync=jnp.zeros(shape=(), dtype=jnp.int32),
            opt_state=optimizer.init(params),
            adv_state=adv_optimizer.init(params),
            cache=None if opaque_mode else params,
        )

    def opaque_update_fn(
            updates: base.Updates,
            state: SAMState,
            params: Optional[base.Params],
            *,
            grad_fn: Optional[Callable[[base.Params, int], base.Updates]] = None,
    ) -> tuple[base.Updates, SAMState]:
        if grad_fn is None:
            raise ValueError("grad_fn must be provided when opaque_mode=True.")

        outer_params = params
        adv_params = params
        adv_updates = updates
        adv_state = state.adv_state

        for i in range(sync_period - 1):
            # Compute second-moment estimate for adaptive scaling
            v_t = jax.tree_map(lambda g: jnp.sqrt(jnp.mean(g ** 2)) + 1e-12, adv_updates)

            # Compute adaptive perturbation
            adv_updates = jax.tree_map(lambda g, v: g / (v + 1e-12), adv_updates, v_t)

            adv_updates, adv_state = adv_optimizer.update(
                adv_updates, adv_state, adv_params
            )
            adv_updates = jax.tree_map(lambda x: -x, adv_updates)

            adv_params = update.apply_updates(adv_params, adv_updates)
            adv_updates = grad_fn(adv_params, i)

            if batch_axis_name is not None:
                adv_updates = jax.lax.pmean(adv_updates, axis_name=batch_axis_name)

        if reset_state:
            adv_state = adv_optimizer.init(outer_params)

        updates, opt_state = optimizer.update(
            adv_updates, state.opt_state, outer_params
        )

        return updates, SAMState(
            steps_since_sync=jnp.zeros(shape=(), dtype=jnp.int32),
            adv_state=adv_state,
            opt_state=opt_state,
            cache=None,
        )

    update_fn = opaque_update_fn

    return base.GradientTransformationExtraArgs(init_fn, update_fn)


@chex.dataclass
class LookSAMState(SAMState):
    """State of `GradientTransformation` for LookSAM."""
    avg_grad: base.Updates  # Stores moving average of gradients


def looksam(
        optimizer: base.GradientTransformation,
        adv_optimizer: base.GradientTransformation,
        sync_period: int = 2,
        reset_state: bool = True,
        beta: float = 0.9,  # Moving average factor
        opaque_mode: bool = False,
        batch_axis_name: Optional[str] = None,
) -> base.GradientTransformationExtraArgs:
    if sync_period < 1:
        raise ValueError("Synchronization period must be >= 1.")

    def init_fn(params: base.Params) -> LookSAMState:
        avg_grad = jax.tree_map(jnp.zeros_like, params)
        return LookSAMState(
            steps_since_sync=jnp.zeros(shape=(), dtype=jnp.int32),
            opt_state=optimizer.init(params),
            adv_state=adv_optimizer.init(params),
            cache=None if opaque_mode else params,
            avg_grad=avg_grad,
        )

    def opaque_update_fn(
            updates: base.Updates,
            state: LookSAMState,
            params: Optional[base.Params],
            *,
            grad_fn: Optional[Callable[[base.Params, int], base.Updates]] = None,
    ) -> tuple[base.Updates, LookSAMState]:
        if grad_fn is None:
            raise ValueError("grad_fn must be provided when opaque_mode=True.")

        outer_params = params
        adv_params = params
        adv_updates = updates
        adv_state = state.adv_state
        avg_grad = state.avg_grad
        new_avg_grad = avg_grad

        for i in range(sync_period - 1):
            # Exponential moving average of gradients
            new_avg_grad = jax.tree_map(lambda avg, g: beta * avg + (1 - beta) * g, avg_grad, adv_updates)

            # Compute LookSAM perturbation
            perturbations = jax.tree_map(lambda g: g / (jnp.linalg.norm(g) + 1e-12), new_avg_grad)

            adv_updates, adv_state = adv_optimizer.update(
                perturbations, adv_state, adv_params
            )
            adv_updates = jax.tree_map(lambda x: -x, adv_updates)

            adv_params = update.apply_updates(adv_params, adv_updates)
            adv_updates = grad_fn(adv_params, i)

            if batch_axis_name is not None:
                adv_updates = jax.lax.pmean(adv_updates, axis_name=batch_axis_name)

        if reset_state:
            adv_state = adv_optimizer.init(outer_params)

        updates, opt_state = optimizer.update(
            adv_updates, state.opt_state, outer_params
        )

        return updates, LookSAMState(
            steps_since_sync=jnp.zeros(shape=(), dtype=jnp.int32),
            adv_state=adv_state,
            opt_state=opt_state,
            cache=None,
            avg_grad=new_avg_grad,
        )

    update_fn = opaque_update_fn

    return base.GradientTransformationExtraArgs(init_fn, update_fn)
