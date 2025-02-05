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

def train_model(state, model, loss_fn, metrics_history, n_epochs, loaders, name, callbacks=[],
                tqdm_over_epochs=True, tqdm_over_batch=False, force_fb=False, verbose=False,
                tqdm_mininterval=0.1, tqdm_maxinterval=10.,):
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
    epoch_bar = tqdm(range(1, n_epochs+1), desc="epochs", total=n_epochs, miniters=tqdm_over_epochs,
                     maxinterval=tqdm_maxinterval, mininterval=tqdm_mininterval) if tqdm_over_epochs>0 \
                else range(1, n_epochs+1)
    train_bar = tqdm(train_loader, desc="training", total=len(train_loader)) if tqdm_over_batch else train_loader
    test_bar = tqdm(test_loader, desc="validation", total=len(test_loader)) if tqdm_over_batch else test_loader
    if verbose: print("bar length", len(train_bar), len(test_bar))

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
            return x + y / len(train_bar)

        return jax.tree.map(add, tree_left, tree_right)

    @jax.jit
    def _train_step(state, batch, deterministic=False):
        def get_loss(params):
            preds = state.apply_fn({'params':params}, batch[0], deterministic=deterministic)
            loss = loss_fn(preds, batch[1]).mean()
            return loss
        grad_fn = jax.grad(get_loss)
        grads = grad_fn(state.params)
        state = state.apply_gradients(grads=grads)
        return state

    @jax.jit
    def _get_grads(state, batch, deterministic=False):
        def get_loss(params):
            preds = state.apply_fn({'params':params}, batch[0], deterministic=deterministic)
            loss = loss_fn(preds, batch[1]).mean()
            return loss
        grad_fn = jax.grad(get_loss)
        grads = grad_fn(state.params)
        return grads

    @jax.jit
    def _compute_metrics(*, state, batch, deterministic=True):
        preds = state.apply_fn({'params': state.params}, batch[0], deterministic=deterministic)
        loss = loss_fn(preds, batch[1]).mean()
        metric_updates = state.metrics.single_from_model_output(
            logits=preds, labels=batch[1], loss=loss)
        metrics = state.metrics.merge(metric_updates)
        state = state.replace(metrics=metrics)
        return state

    # compute metrics at epoch 0
    deterministic = False # no BN stats yet
    tmp_state = state
    for batch in train_bar:
        tmp_state = _compute_metrics(state=tmp_state, batch=batch, deterministic=deterministic)
    for metric, value in tmp_state.metrics.compute().items():  # compute metrics
        metrics_history[f'train_{metric}'].append(value)  # record metrics
    for batch in test_bar:
        tmp_state = _compute_metrics(state=tmp_state, batch=batch, deterministic=deterministic)
    for metric, value in tmp_state.metrics.compute().items():  # compute metrics
        metrics_history[f'test_{metric}'].append(value)  # record metrics
    del tmp_state

    # model training
    start_time = time.time()
    for epoch in epoch_bar:
        # ---------------------------------------
        # train
        # ---------------------------------------
        deterministic = False

        if not force_fb or len(train_bar) == 1:
            # running minibatch GD
            for batch in train_bar:
                state = _train_step(state, batch)  # get updated train state (which contains the updated parameters)
                state = _compute_metrics(state=state, batch=batch)  # aggregate batch metrics
            for metric, value in state.metrics.compute().items():  # compute metrics
                metrics_history[f'train_{metric}'].append(value)  # record metrics
            state = state.replace(metrics=state.metrics.empty())  # reset train_metrics for next training epoch

        else:
            grads = _tree_zeros_like(state.params)
            for batch in train_bar:
                batch_grads = _get_grads(state, batch)  # get updated train state (which contains the updated parameters)
                grads = _tree_add(grads, batch_grads)

            state = state.apply_gradients(grads=grads)
            for batch in train_bar:
                state = _compute_metrics(state=state, batch=batch)
            for metric, value in state.metrics.compute().items():  # compute metrics
                metrics_history[f'train_{metric}'].append(value)  # record metrics
            state = state.replace(metrics=state.metrics.empty())  # reset train_metrics for next training epoch

        # ---------------------------------------
        # test
        # ---------------------------------------
        # model = model.eval()
        deterministic = True
        test_state = state
        for test_batch in test_loader:
            test_state = _compute_metrics(state=test_state, batch=test_batch)
        for metric, value in test_state.metrics.compute().items():
            metrics_history[f'test_{metric}'].append(value)

        # ---------------------------------------
        # callbacks
        # ---------------------------------------
        for cb in callbacks:
            try:
                signal = cb.forward(epoch=epoch, state=state, deterministic=deterministic,
                                    tr_acc=metrics_history[f'train_accuracy'][-1],
                                    tr_loss=metrics_history[f'train_loss'][-1])

            except ArithmeticError:
                # in case early stop CB has a divergence
                callback_break_flag=True
                callbacks[-1].final_state = state
                callbacks[-1].final_epoch = epoch
                callbacks[-1].save(error=True) # make sure early stop is the last CB
            if signal == 'break':
                callback_break_flag=True

        # ---------------------------------------
        # logs
        # ---------------------------------------
        bar_text = f"{epoch}"
        try:
            if epoch % tqdm_over_epochs == 0:
                try:
                    tr_acc = metrics_history[f'train_accuracy'][-1]
                    tr_loss = metrics_history[f'train_loss'][-1]
                    te_acc = metrics_history[f'test_accuracy'][-1]
                    te_loss = metrics_history[f'test_loss'][-1]
                    bar_text = f"{epoch}, train:{tr_loss:.2E}/{tr_acc*100:.0%}; test:{te_loss:.2E}/{te_acc*100:.0%}"
                    epoch_bar.set_description(bar_text)
                except KeyError:
                    tr_loss = metrics_history[f'train_loss'][-1]
                    te_loss = metrics_history[f'test_loss'][-1]
                    bar_text = f"{epoch}, train:{tr_loss:.2E}; test:{te_loss:.2E}"
                except AttributeError:
                    pass
                epoch_bar.set_description(bar_text)
        except ZeroDivisionError:
            pass
        if callback_break_flag:
            print("terminating training", bar_text)
            break

    # save histories
    utils.save_thing(metrics_history, "traj/" + name + "/metrics.pkl")

    for cb in callbacks:
        cb.save()

    return 1


