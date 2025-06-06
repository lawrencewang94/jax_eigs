import jax
import jax.numpy as jnp
import jax.random as rnd
import numpy as np
import optax
from functools import partial
import typing as tp
from tqdm.auto import tqdm
import copy
import os
import jax.flatten_util as fu
import jax.tree_util as tu
import importlib
import time
from optax import contrib

# import refactor.utils as utils
import utils

importlib.reload(utils)
#define jax types
Batch = tp.Mapping[str, np.ndarray]
# Model = tx.Sequential
# Model = to.node
Model = tp.Any
Logs = tp.Dict[str, jnp.ndarray]


def _tree_zeros_like(tree):
    def zeros(x):
        return np.zeros_like(x)
    return tu.tree_map(zeros, tree)


def _get_train_jits(loss_fn, option=""):

    @jax.jit
    def _train_step_vanilla(state, batch):
        rng, do_rng = jax.random.split(state.rng)

        def get_loss(params):
            preds = state.apply_fn({'params': params,},
                                            batch[0], train=True, rngs={"dropout": do_rng})
            loss = loss_fn(preds, batch[1]).mean()
            return loss

        grad_fn = jax.grad(get_loss)
        grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(rng=rng)
        return state

    @jax.jit
    def _get_grads_vanilla(state, batch):
        rng, do_rng = jax.random.split(state.rng)

        def get_loss(params):
            preds = state.apply_fn({'params': params,}, batch[0], train=True, rngs={"dropout": do_rng})
            loss = loss_fn(preds, batch[1]).mean()
            return loss

        grad_fn = jax.grad(get_loss)
        grads = grad_fn(state.params)
        state = state.replace(rng=rng)
        return state, grads

    @jax.jit
    def _compute_metrics_vanilla(*, state, batch):
        preds = state.apply_fn({'params': state.params}, batch[0], train=False)
        loss = loss_fn(preds, batch[1]).mean()
        metric_updates = state.metrics.single_from_model_output(
            logits=preds, labels=batch[1], loss=loss)
        metrics = state.metrics.merge(metric_updates)
        state = state.replace(metrics=metrics)
        return state

    @jax.jit
    def _train_step_bn(state, batch):
        rng, do_rng = jax.random.split(state.rng)

        def get_loss(params):
            preds, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                            batch[0], train=True, mutable=['batch_stats'], rngs={"dropout": do_rng})
            loss = loss_fn(preds, batch[1]).mean()
            return loss, updates

        grad_fn = jax.value_and_grad(get_loss, has_aux=True)
        (loss, updates), grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=updates['batch_stats'], rng=rng)
        return state

    @jax.jit
    def _get_grads_bn(state, batch):
        rng, do_rng = jax.random.split(state.rng)

        def get_loss(params):
            preds, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                            batch[0], train=True, mutable=['batch_stats'], rngs={"dropout": do_rng})
            loss = loss_fn(preds, batch[1]).mean()
            return loss, updates

        grad_fn = jax.value_and_grad(get_loss, has_aux=True)
        (loss, updates), grads = grad_fn(state.params)
        state = state.replace(batch_stats=updates['batch_stats'], rng=rng)
        return state, grads

    @jax.jit
    def _compute_metrics_bn(*, state, batch):
        preds = state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats}, batch[0], train=False)
        loss = loss_fn(preds, batch[1]).mean()
        metric_updates = state.metrics.single_from_model_output(
            logits=preds, labels=batch[1], loss=loss)
        metrics = state.metrics.merge(metric_updates)
        state = state.replace(metrics=metrics)
        return state

    @jax.jit
    def _train_step_sam(state, batch):
        rng, do_rng = jax.random.split(state.rng)

        def get_loss(params):
            preds, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                            batch[0], train=True, mutable=['batch_stats'], rngs={"dropout": do_rng})
            loss = loss_fn(preds, batch[1]).mean()
            return loss, updates

        def get_loss_adv(params):
            preds, _ = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                            batch[0], train=True, mutable=['batch_stats'], rngs={"dropout": do_rng})
            loss = loss_fn(preds, batch[1]).mean()
            return loss

        grad_fn = jax.value_and_grad(get_loss, has_aux=True)
        (loss, updates), grads = grad_fn(state.params)

        if isinstance(state.opt_state, contrib.SAMState):
            state = state.apply_gradients_SAM(grads=grads, loss_wrap=get_loss_adv)
        else:
            state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=updates['batch_stats'], rng=rng)
        return state

    @jax.jit
    def _get_grads_sam(state, batch):
        rng, do_rng = jax.random.split(state.rng)

        def get_loss(params):
            preds, updates = state.apply_fn({'params': params, 'batch_stats': state.batch_stats},
                                            batch[0], train=True, mutable=['batch_stats'], rngs={"dropout": do_rng})
            loss = loss_fn(preds, batch[1]).mean()
            return loss, updates

        grad_fn = jax.value_and_grad(get_loss, has_aux=True)
        (loss, updates), grads = grad_fn(state.params)
        state = state.replace(batch_stats=updates['batch_stats'], rng=rng)
        return state, grads

    @jax.jit
    def _compute_metrics_sam(*, state, batch):
        preds = state.apply_fn({'params': state.params, 'batch_stats': state.batch_stats}, batch[0], train=False)
        loss = loss_fn(preds, batch[1]).mean()
        metric_updates = state.metrics.single_from_model_output(
            logits=preds, labels=batch[1], loss=loss)
        metrics = state.metrics.merge(metric_updates)
        state = state.replace(metrics=metrics)
        return state

    if option == "":
        return _train_step_vanilla, _get_grads_vanilla, _compute_metrics_vanilla,
    elif option == "bn":
        return _train_step_bn, _get_grads_bn, _compute_metrics_bn,
    elif option == "sam":
        return _train_step_sam, _get_grads_sam, _compute_metrics_sam,
    else:
        raise NotImplementedError


def train_model(state, model, loss_fn, metrics_history, n_epochs, loaders, name, callbacks=[], option="",
                tqdm_over_epochs=True, tqdm_over_batch=False, force_fb=False, verbose=False, cfg=None, dynamic_bar=True,
                tqdm_maxinterval=1800, eval_freq=1, gradient_accumulation=1, return_state=False, steps_not_epochs=False):
    '''
    # check stuff, define stuff, make folders,
    # set up loops
    # initialize callbacks
    # add a blank step on both train and val, run callbacks once
    # have different loops for if force FB or not, run train loop, run eval loop at eval freq,
    # run all callbacks, check for callback termination signal
    # final evaluation on train and test. #
    # Deterministic argument in flax https://flax-linen.readthedocs.io/en/latest/guides/flax_fundamentals/arguments.html
    # save things and be done
    '''

    # define things
    assert len(loaders) == 2
    print("Training model", name)
    train_loader, test_loader = loaders[0], loaders[1]
    # make the traj folder
    if not os.path.exists("traj/"+name):
        os.mkdir("traj/"+name)

    # set up tqdm
    if isinstance(tqdm_over_epochs, bool):
        tqdm_over_epochs = int(tqdm_over_epochs)

    if tqdm_over_epochs > 0:
        if dynamic_bar:
            epoch_bar = tqdm(range(1, n_epochs + 1), desc="epochs", total=n_epochs, maxinterval=tqdm_maxinterval,
                         miniters=tqdm_over_epochs)
        else:
            epoch_bar = tqdm(range(1, n_epochs + 1), desc="epochs", total=n_epochs, maxinterval=tqdm_maxinterval,
                             miniters=tqdm_over_epochs, disable=True)
    else:
        epoch_bar = range(1, n_epochs+1)


    train_bar = tqdm(train_loader, desc="training", total=len(train_loader)) if tqdm_over_batch else train_loader
    test_bar = tqdm(test_loader, desc="validation", total=len(test_loader)) if tqdm_over_batch else test_loader
    if verbose: print("bar length", len(train_bar), len(test_bar))

    # set up grad_accum
    if force_fb:
        n_grad_accum = len(train_bar)
    else:
        n_grad_accum = gradient_accumulation

# define callbacks
    callback_break_flag = False
    if len(callbacks) > 0:
        cb_name = utils.get_callback_name_str(callbacks)
        utils.save_thing(cb_name, f"traj/{name}/callbacks.pkl")

    for cb in callbacks:
        cb.init(epoch=0, model=model, state=state, exp_name=name)
        cb.forward(epoch=0, model=model, state=state)

    # define helper functions
    @jax.jit
    def _tree_add(tree_left, tree_right):
        """Computes tree_left + tree_right."""
        def add(x, y):
            return x + y

        return jax.tree.map(add, tree_left, tree_right)

    @jax.jit
    def _tree_div(tree, divisor):
        return jax.tree_map(lambda x: x / divisor, tree)

    _train_step, _get_grads, _compute_metrics = _get_train_jits(loss_fn, option=option)

    # compute metrics at epoch 0
    if steps_not_epochs:
        train_iter = iter(train_loader)
        batch = next(train_iter)
        state = _compute_metrics(state=state, batch=batch)
    else:
        for batch in train_bar:
            state = _compute_metrics(state=state, batch=batch)
    for metric, value in state.metrics.compute().items():  # compute metrics
        metrics_history[f'train_{metric}'].append(value)  # record metrics
    state = state.replace(metrics=state.metrics.empty())  # reset train_metrics for next training epoch

    for batch in test_bar:
        state = _compute_metrics(state=state, batch=batch)
    for metric, value in state.metrics.compute().items():  # compute metrics
        metrics_history[f'test_{metric}'].append(value)  # record metrics
    state = state.replace(metrics=state.metrics.empty())  # reset train_metrics for next training epoch

    if 'grad_norm' not in metrics_history.keys():
        metrics_history['grad_norm'] = []
    # model training
    start_time = time.time()
    epoch = 0


    def set_bar_text(bar_text):
        if tqdm_over_epochs > 0:
            if dynamic_bar:
                epoch_bar.set_description(bar_text)
            else:
                total = n_epochs
                current_time = time.time() - start_time
                per_unit_time = current_time / (epoch + 1)
                remaining = n_epochs - epoch
                eta = remaining * per_unit_time
                out_string = f"GPU {cfg.gpu_id} | Epoch {epoch + 1}/{total} | ETA: {eta:.1f}s" + bar_text
                print(out_string)
        else:
            total = n_epochs
            current_time = time.time()-start_time
            per_unit_time = current_time / (epoch + 1)
            remaining = n_epochs - epoch
            eta = remaining * per_unit_time
            out_string = f"GPU {cfg.gpu_id} | Epoch {epoch + 1}/{total} | ETA: {eta:.1f}s" + bar_text
            print(out_string)

    print("Beginning training")

    for epoch in epoch_bar:
        # initialize bar text
        if epoch == 0:
            bar_text = utils.compute_bar_text(metrics_history, epoch)
            # epoch_bar.set_description(bar_text)
            set_bar_text(bar_text)

        # ---------------------------------------
        # train by timesteps
        # ---------------------------------------
        if steps_not_epochs:

            # get batch
            try:
                batch = next(train_iter)
            except StopIteration:
                # Reload buffer if using a streaming loader
                train_iter = iter(train_loader)
                batch = next(train_iter)

            # compute gradients
            if n_grad_accum == 1:
                state = _train_step(state, batch)
                state = _compute_metrics(state=state, batch=batch)
            else:
                accum_counter = 0
                grads = _tree_zeros_like(state.params)

                while accum_counter < n_grad_accum:
                    try:
                        batch = next(train_iter)
                        state, batch_grads = _get_grads(state, batch)  # get updated train state (which contains the updated parameters)
                        grads = _tree_add(grads, batch_grads)
                        accum_counter += 1

                        state = _compute_metrics(state=state, batch=batch)

                    except StopIteration:
                        # Reload buffer if using a streaming loader
                        train_iter = iter(train_loader)

                if accum_counter == n_grad_accum:
                    grads = _tree_div(grads, accum_counter)
                    state = state.apply_gradients(grads=grads)
                    metrics_history['grad_norm'].append(optax.global_norm(grads))

        # ---------------------------------------
        # train by epochs
        # ---------------------------------------
        else:
            if n_grad_accum == 1:
                # running minibatch GD
                for batch in train_bar:
                    state = _train_step(state, batch)  # get updated train state (which contains the updated parameters)
                    state = _compute_metrics(state=state, batch=batch)  # aggregate batch metrics
            else:
                grads = _tree_zeros_like(state.params)
                accum_counter = 0

                for batch in train_bar:
                    state, batch_grads = _get_grads(state, batch)  # get updated train state (which contains the updated parameters)
                    grads = _tree_add(grads, batch_grads)
                    accum_counter += 1
                    state = _compute_metrics(state=state, batch=batch)

                    if accum_counter == n_grad_accum:
                        grads = _tree_div(grads, accum_counter)
                        state = state.apply_gradients(grads=grads)
                        grads = _tree_zeros_like(state.params)
                        accum_counter = 0

                # apply leftover gradients
                if accum_counter > 0:
                    grads = _tree_div(grads, accum_counter)
                    state = state.apply_gradients(grads=grads)

        # ---------------------------------------
        # compute training metrics
        # ---------------------------------------
        for metric, value in state.metrics.compute().items():  # compute metrics
            metrics_history[f'train_{metric}'].append(value)  # record metrics
        state = state.replace(metrics=state.metrics.empty())  # reset train_metrics for next training epoch

        # ---------------------------------------
        # perform callbacks
        # ---------------------------------------
        train = False
        if epoch % eval_freq == 0:
            for cb in callbacks:
                try:
                    signal = cb.forward(epoch=epoch, state=state, train=train,
                                        mh=metrics_history,)

                except ArithmeticError:
                    # in case early stop CB has a divergence
                    print("ES Arithmetic Error")
                    callback_break_flag=True
                    callbacks[-1].final_state = state
                    callbacks[-1].final_epoch = epoch
                    callbacks[-1].save(error=True) # make sure early stop is the last CB
                if signal == 'break':
                    callback_break_flag=True

        # ---------------------------------------
        # evaluate on test data
        # ---------------------------------------
        # model = model.eval()
        # evaluate every N epochs/steps
        if epoch % eval_freq == 0 or callback_break_flag:
            for test_batch in test_loader:
                state = _compute_metrics(state=state, batch=test_batch)
            for metric, value in state.metrics.compute().items():
                metrics_history[f'test_{metric}'].append(value)
            state = state.replace(metrics=state.metrics.empty())  # reset train_metrics for next training epoch

        # ---------------------------------------
        # logs
        # ---------------------------------------
        bar_text = utils.compute_bar_text(metrics_history, epoch)
        if (tqdm_over_epochs > 0) and ((epoch % tqdm_over_epochs) == 0) or callback_break_flag:
            # epoch_bar.set_description(bar_text)
            set_bar_text(bar_text)

        if callback_break_flag:
            print("terminating training", bar_text)
            break

    if not callback_break_flag and len(callbacks) > 0:
        callbacks[-1].final_state = state
        callbacks[-1].final_epoch = epoch

    metrics_history['lse'] = epoch

    # save histories
    utils.save_thing(metrics_history, "traj/" + name + "/metrics.pkl")

    for cb in callbacks:
        cb.save()

    print("Training complete", name)
    if not return_state:
        return metrics_history
    else:
        return metrics_history, state


