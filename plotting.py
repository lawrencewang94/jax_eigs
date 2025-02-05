import jax
import jax.numpy as jnp
import jax.random as rnd
from jax import flatten_util as fu
from jax import tree_util as tu
import torch
import numpy as np
import optax
import matplotlib.pyplot as plt
import copy
import matplotlib.colors as mcol
import matplotlib.cm as cm
import matplotlib
import seaborn as sns
import scipy
import glob
import re

# import refactor.utils as utils
# import refactor.training as training
import utils
import training

import importlib
importlib.reload(utils)
from utils import dark_colours, light_colours, all_colours

import matplotlib
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import cm
import warnings
warnings.filterwarnings( "ignore", module = "matplotlib" )

default_size = (12, 9)
log_scale_limit = 3.0 * 10e2
log_kwargs = {'base':2}
symlog_kwargs = {'base':10, 'linthresh':1e-1, 'linscale':0.5}
# symlog_scale = lambda axis: matplotlib.scale.SymmetricalLogScale(axis, base=2, linthresh=0.001, subs=None, linscale=1)


def switch_ltype(line, ax):
    x, y, ltype, properties = line
    if ltype == 'scatter':
        l = ax.scatter(x, y, **properties)
    elif ltype == 'line':
        l = ax.plot(x, y, **properties)
    elif ltype == 'vline':
        l = ax.axvline(x, **properties)
    else:
        raise NotImplementedError
    return l

def plot_eigv_density(grids, density, label=None):
    plt.semilogy(grids, density, label=label)
    plt.ylim(1e-10, 1e2)
    plt.ylabel("Density")
    plt.xlabel("Eigenvalue")
    plt.legend()
    return plt


def plot_xy(x, y, scale='linear', title=None, savepath=None, fig=None, ax=None, close=False):
    if fig is None:
        assert ax is None
        fig, ax = plt.subplots(1, 1, figsize=default_size)

    ax.plot(x, y)
    ax.set_yscale(scale)
    ax.set_title(title)
    # plt.show()
    if savepath is not None:
        plt.savefig(savepath, facecolor="w")
    if fig is None and ax is None and close:
        plt.close()


def plot_things(lines, scale='linear', title=None, savepath=None, fig=None, ax=None, label=True, ylim=None, close=False, vlines=None):
    '''
    plots multiple objects on same axis
    lines: x, y, ltype, properties
    '''
    if fig is None:
        assert ax is None
        fig, ax = plt.subplots(1, 1, figsize=default_size)

    for i, line in enumerate(lines):
        l = switch_ltype(line, ax)

    if label: ax.legend()
    if vlines is not None:
        for xc in vlines:
            plt.axvline(x=xc, ls='--', c='k', label="_nolegend_")
    # ax.set_yscale(scale)
    if ax.get_ylim()[0] < -log_scale_limit or ax.get_ylim()[1] > log_scale_limit:
        ax.set_yscale('log', **log_kwargs)
    else:
        ax.set_yscale(scale)
    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])

    if savepath is not None:
        plt.savefig(savepath, facecolor="w")
    if fig is None and ax is None and close:
        plt.close()


def plot_things_dual_axes(lines, lines2, scale='linear', scale2="linear", title=None, savepath=None, fig=None, ax=None, label=True, close=False, vlines=None):

    '''
    plots multiple objects on same axis
    lines: x, y, ltype, properties
    '''

    if fig is None:
        assert ax is None
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=default_size)

    ax2 = ax.twinx()
    ls = []
    for line in lines:
        l = switch_ltype(line, ax)
        ls.append(l[0])

    l2s = []
    for line in lines2:
        l = switch_ltype(line, ax2)
        l2s.append(l[0])

    if label:
        lns = ls + l2s
        labs = [l.get_label() for l in ls] + [l.get_label() for l in l2s]
        ax.legend(lns, labs, loc=0)

    if vlines is not None:
        for xc in vlines:
            ax.axvline(x=xc, ls='--', c='k', label="_nolegend_")
    # ax.set_yscale(scale)
    if ax.get_ylim()[0] < -log_scale_limit or ax.get_ylim()[1] > log_scale_limit:
        ax.set_yscale('log', **log_kwargs)
    else:
        ax.set_yscale(scale)

    # ax2.set_yscale(scale2)
    if ax2.get_ylim()[0] < -log_scale_limit or ax2.get_ylim()[1] > log_scale_limit:
        ax2.set_yscale('log', **log_kwargs)
    else:
        ax2.set_yscale(scale)
    ax.set_title(title)

    if savepath is not None:
        plt.savefig(savepath, facecolor="w")
    if fig is None and ax is None and close:
        plt.close()


def plot_things_dual_axes_sharefix(lines, lines2, scale='linear', scale2="linear", title=None, savepath=None, fig=None, ax=None, label=True, close=False):
    '''
    plots multiple objects on same axis
    lines: x, y, ltype, properties
    '''
    if fig is None:
        assert ax is None
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=default_size)

    for line in lines:
        l = switch_ltype(line, ax)

    ylims0 = ax.get_ylim()

    for line in lines2:
        l = switch_ltype(line, ax)

    # ylims1 = ax.get_ylim()
    # ylims = (min(ylims0[0], ylims1[0]), max(ylims0[1], ylims1[1]))
    if label: ax.legend()
    ax.set_yscale(scale)
    ax.set_ylim(ylims0)
    # ax2.set_yscale(scale2)
    ax.set_title(title)

    if savepath is not None:
        plt.savefig(savepath, facecolor="w")
    if fig is None and ax is None and close:
        plt.close()


def plot_eig_evo(eigs, eos=None, neff=None, neff_c=None, title=None, savepath=None, fig=None, ax=None, close=False, scale='linear', xvs=[], vis_n=False):
    # print("n_xs", eigs.shape)
    if fig is None:
        assert ax is None
        # fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

    # this_cmaps = [matplotlib.colors.LinearSegmentedColormap.from_list("", [light_colours(0), dark_colours(0)])]
    n_plots = eigs.shape[0]
    n_eigs = eigs.shape[1]
    # xs = np.array(range(eigs.shape[0]))[:, np.newaxis]
    # xs = np.repeat(xs, n_eigs, axis=1)
    cmap = matplotlib.cm.get_cmap('viridis')
    c_norm = matplotlib.colors.Normalize(vmin=0, vmax=n_eigs)

    xs = np.array(range(eigs.shape[0]))[:, np.newaxis]
    ls = []
    l2s = []
    for i in range(n_eigs):
        ax.plot(xs, eigs[:, i], linewidth=0.5, color=cmap(c_norm(i)))
    if eos:
        ls.append(ax.plot(xs, np.ones(xs.shape[0]) * eos, label="EoS", c='k', alpha=0.5, linestyle="--")[0])

    if vis_n:
        ax2 = ax.twinx()
        ax2.set_ylabel("Neff")

    if neff:
        ls.append(ax.plot(xs, np.max(eigs, axis=1) * neff, label="SANE reg", c=dark_colours(0), linestyle="--")[0])

        if vis_n:
            neff_values = utils.compute_neff(eigs, neff, abs=True)
            l2s.append(ax2.plot(xs, neff_values, c=dark_colours(0), label='SANE')[0])

    if neff_c:
        ls.append(ax.plot(xs, np.ones(len(eigs))*neff_c, label="Neff_c reg", c=dark_colours(1), linestyle="--")[0])
        if vis_n:
            neff_values = utils.compute_neff_const(eigs, neff, abs=True)
            l2s.append(ax2.plot(xs, neff_values, c=dark_colours(1), label='neff_c')[0])


    # ls.append(ax.plot(xs, np.zeros(xs.shape[0]), label="Zero", linestyle="--")[0])
    min_line = np.min(eigs, axis=1)
    ls.append(ax.plot(xs, min_line, label="HessVar", linestyle="--", c='r')[0])
    ls.append(ax.plot(xs, -min_line, label="_nolegend_", linestyle="--", c='r')[0])
    # max_line = np.max(eigs, axis=1)
    ax.set_title(title)
    if ax.get_ylim()[0] < -log_scale_limit or ax.get_ylim()[1] > log_scale_limit:
        ax.set_yscale('symlog', **symlog_kwargs)
    else:
        ax.set_yscale(scale)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("λ")

    if vis_n:
        lns = ls + l2s
        labs = [l.get_label() for l in ls] + [l.get_label() for l in l2s]
        ax.legend(lns, labs, loc=0)
    else:
        plt.legend()
    for xv in xvs:
        ax.axvline(x=xv, c='k')
    # plt.legend()
    # plt.show()
    if savepath is not None:
        plt.savefig(savepath, facecolor="w")
    if fig is None and ax is None and close:
        plt.close()


def make_cmap(c1="r", c2="b", mini=0, maxi=1):
    # Make a user-defined colormap.
    cm1 = mcol.LinearSegmentedColormap.from_list("MyCmapName", [c1, c2])

    # Make a normalizer that will map the time values from
    # [start_time,end_time+1] -> [0,1].
    cnorm = mcol.Normalize(vmin=mini, vmax=maxi)

    # Turn these into an object that can be used to map time values to colors and
    # can be passed to plt.colorbar().
    cpick = cm.ScalarMappable(norm=cnorm, cmap=cm1)
    cpick.set_array([])
    return cpick


def bc_check_neighbours(this_inds, n_points=100):
    out_inds = []
    n_inds = len(this_inds)
    max_inds = n_points ** 2

    def bin_search(arr, low, high, x):
        # Check base case
        if high >= low:
            mid = (high + low) // 2
            # If element is present at the middle itself
            if arr[mid] == x:
                return mid
            # If element is smaller than mid, then it can only
            # be present in left subarray
            elif arr[mid] > x:
                return bin_search(arr, low, mid - 1, x)
            # Else the element can only be present in right subarray
            else:
                return bin_search(arr, mid + 1, high, x)
        else:
            # Element is not present in the array
            return -1

    for ind in this_inds:
        # check up
        if ind >= n_points:
            if bin_search(this_inds, 0, n_inds - 1, ind - n_points) == -1:
                out_inds.append(ind)
                continue
        # check bottom
        if ind < max_inds - n_points:
            if bin_search(this_inds, 0, n_inds - 1, ind + n_points) == -1:
                out_inds.append(ind)
                continue

        # check left
        if ind % n_points > 0:
            if bin_search(this_inds, 0, n_inds - 1, ind - 1) == -1:
                out_inds.append(ind)
                continue

        # check right
        if (ind + 1) % n_points != 0:
            if bin_search(this_inds, 0, n_inds - 1, ind + 1) == -1:
                out_inds.append(ind)
                continue
    return out_inds


def plot_bc_sr(net, batch, x_range, pred_fn=training.predict, mode="bin", binary=None, title=None, savepath=None,
               fig=None, ax=None, close=False):

    if type(x_range) is not list and type(x_range) is not tuple:
        x_range = (-x_range, x_range)
    # test samples, pred, gt, 2 by 1
    inputs, targets = batch
    # fig, ax = plt.subplots(1, 1, sharey=True, figsize=(16, 8))
    if fig is None:
        assert ax is None
        fig, ax = plt.subplots(1, 1, figsize=default_size)


    preds = pred_fn(net, inputs)

    zero_inds = np.where(preds == 0)
    one_inds = np.where(preds == 1)
    zero_inds_gt = np.where(targets == 0)
    one_inds_gt = np.where(targets == 1)

    ax.scatter(inputs[zero_inds_gt, 0], inputs[zero_inds_gt, 1], c="r")
    ax.scatter(inputs[one_inds_gt, 0], inputs[one_inds_gt, 1], c="b")
    tmp_x = torch.linspace(x_range[0], x_range[1], 100)
    tmp_y = torch.linspace(x_range[0], x_range[1], 100)
    grid_x, grid_y = torch.meshgrid(tmp_x, tmp_y)
    grid_x_flat = grid_x.flatten().view(-1, 1)
    grid_y_flat = grid_y.flatten().view(-1, 1)
    xs = np.concatenate([grid_x_flat, grid_y_flat], axis=1)
    grid_output = pred_fn(net, xs)
    if binary is not None and binary: mode = "bin"
    if binary is not None and binary == False: mode = "cont"

    if mode == "bin":
        grid_output_argmax = np.argmax(grid_output, axis=-1)
        neg_inds = np.where(grid_output_argmax == 0)[0]
        pos_inds = np.where(grid_output_argmax == 1)[0]
        ax.scatter(grid_x_flat[neg_inds], grid_y_flat[neg_inds], c="r", alpha=0.2)
        ax.scatter(grid_x_flat[pos_inds], grid_y_flat[pos_inds], c="b", alpha=0.2)
    elif mode == "cont":
        grid_output_softmax = scipy.special.softmax(grid_output, axis=1)
        cpick = make_cmap("r", "b", 0, 1)
        ax.scatter(grid_x_flat, grid_y_flat, c=cpick.to_rgba(grid_output_softmax[:, 1]), alpha=0.2)
        # plt.colorbar(cpick, label="heatmap, 0(r)->1(b)")
    elif mode == "bin_line":
        grid_output_argmax = np.argmax(grid_output, axis=-1)
        neg_inds = np.where(grid_output_argmax == 0)[0]
        pos_inds = np.where(grid_output_argmax == 1)[0]
        print(neg_inds.shape, pos_inds.shape)

        pos_ind_n = bc_check_neighbours(pos_inds)
        neg_ind_n = bc_check_neighbours(neg_inds)
        ax.scatter(grid_x_flat[neg_ind_n], grid_y_flat[neg_ind_n], s=2, c="r", alpha=0.3)
        ax.scatter(grid_x_flat[pos_ind_n], grid_y_flat[pos_ind_n], s=2, c="b", alpha=0.3)
    plt.title(title)
    # plt.show()
    if savepath is not None:
        plt.savefig(savepath, facecolor="w")
    if fig is None and ax is None and close:
        plt.close()


def plot_ablations_sr(net, batch, x_range, eigv, eigv_scale=0.2, pred_fn=training.predict,
                      mode="bin", binary=None, title=None, savepath=None, fig=None, ax=None, close=False):

    if type(x_range) is not list and type(x_range) is not tuple:
        x_range = (-x_range, x_range)

    # test samples, pred, gt, 2 by 1
    if fig is None:
        assert ax is None
        #         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
        fig, ax = plt.subplots(1, 3, sharey=True, figsize=(24, 6))

    inputs, targets = batch

    zero_inds_gt = np.where(targets == 0)
    one_inds_gt = np.where(targets == 1)

    ax[0].scatter(inputs[zero_inds_gt, 0], inputs[zero_inds_gt, 1], c="r")
    ax[0].scatter(inputs[one_inds_gt, 0], inputs[one_inds_gt, 1], c="b")
    ax[1].scatter(inputs[zero_inds_gt, 0], inputs[zero_inds_gt, 1], c="r")
    ax[1].scatter(inputs[one_inds_gt, 0], inputs[one_inds_gt, 1], c="b")
    ax[2].scatter(inputs[zero_inds_gt, 0], inputs[zero_inds_gt, 1], c="r")
    ax[2].scatter(inputs[one_inds_gt, 0], inputs[one_inds_gt, 1], c="b")

    tmp_x = torch.linspace(x_range[0], x_range[1], 100)
    tmp_y = torch.linspace(x_range[0], x_range[1], 100)
    grid_x, grid_y = torch.meshgrid(tmp_x, tmp_y)
    grid_x_flat = grid_x.flatten().view(-1, 1)
    grid_y_flat = grid_y.flatten().view(-1, 1)
    #     xs = torch.cat([grid_x_flat, grid_y_flat], dim=1)
    xs = np.concatenate([grid_x_flat, grid_y_flat], axis=1)

    this_w, model_unravel = fu.ravel_pytree(net)
    pos_w = this_w + eigv * eigv_scale
    neg_w = this_w + -eigv * eigv_scale
    #     pos_w = list_add(this_w, unravel_flat(eigv*eigv_scale, this_w))
    #     neg_w = list_add(this_w, unravel_flat(-eigv*eigv_scale, this_w))
    pos_net = copy.deepcopy(net).merge(model_unravel(pos_w))
    neg_net = copy.deepcopy(net).merge(model_unravel(neg_w))

    grid_output = pred_fn(net, xs)
    pos_output = pred_fn(pos_net, xs)
    neg_output = pred_fn(neg_net, xs)

    if binary is not None and binary: mode = "bin"
    if binary is not None and binary == False: mode = "cont"


    if mode == "bin":

        grid_output_argmax = np.argmax(grid_output, axis=-1)
        pos_output_argmax = np.argmax(pos_output, axis=-1)
        neg_output_argmax = np.argmax(neg_output, axis=-1)

        s = 3
        neg_inds = np.where(grid_output_argmax == 0)[0]
        pos_inds = np.where(grid_output_argmax == 1)[0]
        ax[0].scatter(grid_x_flat[neg_inds], grid_y_flat[neg_inds], s=3 * s, c="r", alpha=0.2)
        ax[0].scatter(grid_x_flat[pos_inds], grid_y_flat[pos_inds], s=3 * s, c="b", alpha=0.2)
        pneg_inds = np.where(pos_output_argmax == 0)[0]
        ppos_inds = np.where(pos_output_argmax == 1)[0]
        nneg_inds = np.where(neg_output_argmax == 0)[0]
        npos_inds = np.where(neg_output_argmax == 1)[0]
        ax[1].scatter(grid_x_flat[pneg_inds], grid_y_flat[pneg_inds], s=3 * s, c="r", alpha=0.2)
        ax[1].scatter(grid_x_flat[ppos_inds], grid_y_flat[ppos_inds], s=3 * s, c="b", alpha=0.2)
        ax[2].scatter(grid_x_flat[nneg_inds], grid_y_flat[nneg_inds], s=3 * s, c="r", alpha=0.2)
        ax[2].scatter(grid_x_flat[npos_inds], grid_y_flat[npos_inds], s=3 * s, c="b", alpha=0.2)
        # print(len(neg_inds[0]), len(pneg_inds[0]), len(nneg_inds[0]))

    elif mode == "bin_line":

        grid_output_argmax = np.argmax(grid_output, axis=-1)
        pos_output_argmax = np.argmax(pos_output, axis=-1)
        neg_output_argmax = np.argmax(neg_output, axis=-1)

        s = 3
        neg_inds = np.where(grid_output_argmax == 0)[0]
        pos_inds = np.where(grid_output_argmax == 1)[0]
        neg_inds_n = bc_check_neighbours(neg_inds)
        pos_inds_n = bc_check_neighbours(pos_inds)
        ax[0].scatter(grid_x_flat[neg_inds_n], grid_y_flat[neg_inds_n], s=3 * s, c="r", alpha=0.2)
        ax[0].scatter(grid_x_flat[pos_inds_n], grid_y_flat[pos_inds_n], s=3 * s, c="b", alpha=0.2)
        pneg_inds = np.where(pos_output_argmax == 0)[0]
        ppos_inds = np.where(pos_output_argmax == 1)[0]
        pneg_inds_n = bc_check_neighbours(pneg_inds)
        ppos_inds_n = bc_check_neighbours(ppos_inds)
        ax[1].scatter(grid_x_flat[pneg_inds_n], grid_y_flat[pneg_inds_n], s=3 * s, c="r", alpha=0.2)
        ax[1].scatter(grid_x_flat[ppos_inds_n], grid_y_flat[ppos_inds_n], s=3 * s, c="b", alpha=0.2)
        nneg_inds = np.where(neg_output_argmax == 0)[0]
        npos_inds = np.where(neg_output_argmax == 1)[0]
        nneg_inds_n = bc_check_neighbours(nneg_inds)
        npos_inds_n = bc_check_neighbours(npos_inds)
        ax[2].scatter(grid_x_flat[nneg_inds_n], grid_y_flat[nneg_inds_n], s=3 * s, c="r", alpha=0.2)
        ax[2].scatter(grid_x_flat[npos_inds_n], grid_y_flat[npos_inds_n], s=3 * s, c="b", alpha=0.2)
        # print(len(neg_inds[0]), len(pneg_inds[0]), len(nneg_inds[0]))

    elif mode == "cont":
        grid_output_softmax = scipy.special.softmax(grid_output, axis=-1)
        pos_output_softmax = scipy.special.softmax(pos_output, axis=-1)
        neg_output_softmax = scipy.special.softmax(neg_output, axis=-1)
        cpick = make_cmap("r", "b", 0, 1)
        ax[0].scatter(grid_x_flat, grid_y_flat, c=cpick.to_rgba(grid_output_softmax[:, 1]), alpha=0.1)
        ax[1].scatter(grid_x_flat, grid_y_flat, c=cpick.to_rgba(pos_output_softmax[:, 1]), alpha=0.1)
        ax[2].scatter(grid_x_flat, grid_y_flat, c=cpick.to_rgba(neg_output_softmax[:, 1]), alpha=0.1)

        plt.colorbar(cpick, label="heatmap, 0(r)->1(b)", ax=ax[0])

    ax[0].set_title(title)
    ax[1].set_title("+ve Ablation")
    ax[2].set_title("-ve Ablation")
    # plt.show()
    if savepath is not None:
        plt.savefig(savepath, facecolor="w")
    if fig is None and ax is None and close:
        plt.close()


def plot_ablations_reg(model, batch, x_range, evec, p_scale=0.1, pred_fn=training.pred_normal,
                       fig=None, ax=None, legend=False,
                       title=None, savepath=None, close=False):
    if type(x_range) is not list and type(x_range) is not tuple:
        x_range = (-x_range, x_range)

    if fig is None:
        assert ax is None
        #         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
        fig, ax = plt.subplots(1, 1, sharey=True, figsize=default_size)

    #     ax2 = ax.twinx()

    inputs, targets = batch

    sort_ind = np.argsort(inputs.flatten())
    inputs = inputs[sort_ind, :]
    targets = targets[sort_ind, :]

    xs = inputs

    this_w, model_unravel = fu.ravel_pytree(model)
    pos_w = this_w + evec * p_scale
    neg_w = this_w + -evec * p_scale
    pos_net = model_unravel(pos_w)
    neg_net = model_unravel(neg_w)

    preds = pred_fn(model, inputs)
    pos_preds = pred_fn(pos_net, inputs)
    neg_preds = pred_fn(neg_net, inputs)
    # print(pos_preds)


    ax.plot(xs, preds, label="preds", c=dark_colours(2))
    # ax.plot(xs, pos_preds, label="+ve perturb")
    # ax.plot(xs, neg_preds, label="-ve perturb")
    low_preds = []
    high_preds = []
    # print(pos_preds.shape, neg_preds.shape)
    for i in range(len(pos_preds)):
        low_preds.append(min(pos_preds[i][0], neg_preds[i][0]))
        high_preds.append(max(pos_preds[i][0], neg_preds[i][0]))
    low_preds = np.array(low_preds).flatten()
    high_preds = np.array(high_preds).flatten()
    # print(low_preds.shape)
    # print(xs.shape)
    ax.fill_between(xs.flatten(), low_preds, high_preds, color=dark_colours(2), alpha=0.2)
    ax.scatter(xs, targets, label="target", c='r', s=2)
    # ax.plot(xs, targets, label="target", c='r')

    if legend:
        ax.legend()
    # plt.show()
    ax.set_title(title, fontsize=16)
    if savepath is not None:
        plt.savefig(savepath, facecolor="w")
    if fig is None and ax is None and close:
        plt.close()


def plot_ablations_sr_gni(net, batch, x_range, eigv, noises, error_bar=1, pred_fn=training.predict, n_grid=50,
                      mode="bin", binary=None, title=None, savepath=None, fig=None, ax=None, close=False):

    if type(x_range) is not list and type(x_range) is not tuple:
        x_range = (-x_range, x_range)
    n_samples = len(noises)
    # test samples, pred, gt, 2 by 1
    if fig is None:
        assert ax is None
        #         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
        fig, ax = plt.subplots(1, 3, sharey=True, figsize=(24, 6))

    inputs, targets = batch

    zero_inds_gt = np.where(targets == 0)
    one_inds_gt = np.where(targets == 1)

    ax[0].scatter(inputs[zero_inds_gt, 0], inputs[zero_inds_gt, 1], c="r")
    ax[0].scatter(inputs[one_inds_gt, 0], inputs[one_inds_gt, 1], c="b")
    ax[1].scatter(inputs[zero_inds_gt, 0], inputs[zero_inds_gt, 1], c="r")
    ax[1].scatter(inputs[one_inds_gt, 0], inputs[one_inds_gt, 1], c="b")
    ax[2].scatter(inputs[zero_inds_gt, 0], inputs[zero_inds_gt, 1], c="r")
    ax[2].scatter(inputs[one_inds_gt, 0], inputs[one_inds_gt, 1], c="b")

    tmp_x = torch.linspace(x_range[0], x_range[1], n_grid)
    tmp_y = torch.linspace(x_range[0], x_range[1], n_grid)
    grid_x, grid_y = torch.meshgrid(tmp_x, tmp_y)
    grid_x_flat = grid_x.flatten().view(-1, 1)
    grid_y_flat = grid_y.flatten().view(-1, 1)
    #     xs = torch.cat([grid_x_flat, grid_y_flat], dim=1)
    xs = np.concatenate([grid_x_flat, grid_y_flat], axis=1)

    this_w, model_unravel = fu.ravel_pytree(net)

    grid_output = pred_fn(net, xs)

    pert_outputs = []
    for i in range(n_samples):
        p_w = this_w+noises[i]*eigv
        p_net = copy.deepcopy(net).merge(model_unravel(p_w))
        p_output = pred_fn(p_net, xs)
        pert_outputs.append(p_output)

    pert_outputs = np.array(pert_outputs)
    pert_outputs = np.swapaxes(pert_outputs, 0, 1)
    sorted_p_outputs = np.array(pert_outputs).copy()
    p_outputs_softmax = scipy.special.softmax(pert_outputs, axis=-1)
    p_outputs_inds = np.argsort(p_outputs_softmax[:, :, 0], axis=1)

    sorted_p_outputs = sorted_p_outputs[np.arange(n_grid*n_grid)[:, None], p_outputs_inds]
    pos_output = sorted_p_outputs[:, n_samples-1-error_bar]
    neg_output = sorted_p_outputs[:, error_bar]

    # pos_w = this_w + eigv * eigv_scale
    # neg_w = this_w + -eigv * eigv_scale
    # #     pos_w = list_add(this_w, unravel_flat(eigv*eigv_scale, this_w))
    # #     neg_w = list_add(this_w, unravel_flat(-eigv*eigv_scale, this_w))
    # pos_net = copy.deepcopy(net).merge(model_unravel(pos_w))
    # neg_net = copy.deepcopy(net).merge(model_unravel(neg_w))

    # pos_output = pred_fn(pos_net, xs)
    # neg_output = pred_fn(neg_net, xs)

    if binary is not None and binary: mode = "bin"
    if binary is not None and binary == False: mode = "cont"

    if mode == "bin":

        grid_output_argmax = np.argmax(grid_output, axis=-1)
        pos_output_argmax = np.argmax(pos_output, axis=-1)
        neg_output_argmax = np.argmax(neg_output, axis=-1)

        s = 3
        neg_inds = np.where(grid_output_argmax == 0)[0]
        pos_inds = np.where(grid_output_argmax == 1)[0]
        ax[0].scatter(grid_x_flat[neg_inds], grid_y_flat[neg_inds], s=3 * s, c="r", alpha=0.2)
        ax[0].scatter(grid_x_flat[pos_inds], grid_y_flat[pos_inds], s=3 * s, c="b", alpha=0.2)
        pneg_inds = np.where(pos_output_argmax == 0)[0]
        ppos_inds = np.where(pos_output_argmax == 1)[0]
        nneg_inds = np.where(neg_output_argmax == 0)[0]
        npos_inds = np.where(neg_output_argmax == 1)[0]
        ax[1].scatter(grid_x_flat[pneg_inds], grid_y_flat[pneg_inds], s=3 * s, c="r", alpha=0.2)
        ax[1].scatter(grid_x_flat[ppos_inds], grid_y_flat[ppos_inds], s=3 * s, c="b", alpha=0.2)
        ax[2].scatter(grid_x_flat[nneg_inds], grid_y_flat[nneg_inds], s=3 * s, c="r", alpha=0.2)
        ax[2].scatter(grid_x_flat[npos_inds], grid_y_flat[npos_inds], s=3 * s, c="b", alpha=0.2)
        # print(len(neg_inds[0]), len(pneg_inds[0]), len(nneg_inds[0]))

    elif mode == "bin_line":

        grid_output_argmax = np.argmax(grid_output, axis=-1)
        pos_output_argmax = np.argmax(pos_output, axis=-1)
        neg_output_argmax = np.argmax(neg_output, axis=-1)

        s = 3
        neg_inds = np.where(grid_output_argmax == 0)[0]
        pos_inds = np.where(grid_output_argmax == 1)[0]
        neg_inds_n = bc_check_neighbours(neg_inds)
        pos_inds_n = bc_check_neighbours(pos_inds)
        ax[0].scatter(grid_x_flat[neg_inds_n], grid_y_flat[neg_inds_n], s=3 * s, c="r", alpha=0.2)
        ax[0].scatter(grid_x_flat[pos_inds_n], grid_y_flat[pos_inds_n], s=3 * s, c="b", alpha=0.2)
        pneg_inds = np.where(pos_output_argmax == 0)[0]
        ppos_inds = np.where(pos_output_argmax == 1)[0]
        pneg_inds_n = bc_check_neighbours(pneg_inds)
        ppos_inds_n = bc_check_neighbours(ppos_inds)
        ax[1].scatter(grid_x_flat[pneg_inds_n], grid_y_flat[pneg_inds_n], s=3 * s, c="r", alpha=0.2)
        ax[1].scatter(grid_x_flat[ppos_inds_n], grid_y_flat[ppos_inds_n], s=3 * s, c="b", alpha=0.2)
        nneg_inds = np.where(neg_output_argmax == 0)[0]
        npos_inds = np.where(neg_output_argmax == 1)[0]
        nneg_inds_n = bc_check_neighbours(nneg_inds)
        npos_inds_n = bc_check_neighbours(npos_inds)
        ax[2].scatter(grid_x_flat[nneg_inds_n], grid_y_flat[nneg_inds_n], s=3 * s, c="r", alpha=0.2)
        ax[2].scatter(grid_x_flat[npos_inds_n], grid_y_flat[npos_inds_n], s=3 * s, c="b", alpha=0.2)
        # print(len(neg_inds[0]), len(pneg_inds[0]), len(nneg_inds[0]))

    elif mode == "cont":
        grid_output_softmax = scipy.special.softmax(grid_output, axis=-1)
        pos_output_softmax = scipy.special.softmax(pos_output, axis=-1)
        neg_output_softmax = scipy.special.softmax(neg_output, axis=-1)
        cpick = make_cmap("r", "b", 0, 1)
        ax[0].scatter(grid_x_flat, grid_y_flat, c=cpick.to_rgba(grid_output_softmax[:, 1]), alpha=0.1)
        ax[1].scatter(grid_x_flat, grid_y_flat, c=cpick.to_rgba(pos_output_softmax[:, 1]), alpha=0.1)
        ax[2].scatter(grid_x_flat, grid_y_flat, c=cpick.to_rgba(neg_output_softmax[:, 1]), alpha=0.1)

        plt.colorbar(cpick, label="heatmap, 0(r)->1(b)", ax=ax[0])

    ax[0].set_title(title)
    ax[1].set_title("+ve Ablation")
    ax[2].set_title("-ve Ablation")
    # plt.show()
    if savepath is not None:
        plt.savefig(savepath, facecolor="w")
    if fig is None and ax is None and close:
        plt.close()


def plot_ablations_reg_gni(model, batch, x_range, evec, noises, error_bar=1, pred_fn=training.pred_normal,
                       fig=None, ax=None,
                       title=None, savepath=None, close=False):
    if type(x_range) is not list and type(x_range) is not tuple:
        x_range = (-x_range, x_range)

    if fig is None:
        assert ax is None
        #         fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))
        fig, ax = plt.subplots(1, 1, sharey=True, figsize=default_size)

    #     ax2 = ax.twinx()

    inputs, targets = batch

    sort_ind = np.argsort(inputs.flatten())
    inputs = inputs[sort_ind, :]
    targets = targets[sort_ind, :]

    xs = inputs
    n_samples = len(noises)
    n_outs = len(inputs)

    this_w, model_unravel = fu.ravel_pytree(model)
    preds = pred_fn(model, inputs)

    pert_outputs = []
    for i in range(n_samples):
        p_w = this_w+noises[i]*evec
        p_net = copy.deepcopy(model).merge(model_unravel(p_w))
        p_output = pred_fn(p_net, xs)
        pert_outputs.append(p_output)

    pert_outputs = np.array(pert_outputs)
    pert_outputs = np.swapaxes(pert_outputs, 0, 1)
    sorted_p_outputs = np.array(pert_outputs).copy()
    p_outputs_inds = np.argsort(pert_outputs[:, :, 0], axis=1)
    # print(pert_outputs.shape, sorted_p_outputs.shape, p_outputs_inds.shape)
    sorted_p_outputs = sorted_p_outputs[np.arange(n_outs)[:, None], p_outputs_inds]
    pos_preds = sorted_p_outputs[:, n_samples-1-error_bar]
    neg_preds = sorted_p_outputs[:, error_bar]

    ax.plot(xs, preds, label="preds")
    ax.plot(xs, pos_preds, label="+ve preds")
    ax.plot(xs, neg_preds, label="-ve preds")
    ax.plot(xs, targets, label="target")

    ax.legend()
    # plt.show()
    ax.set_title(title)
    if savepath is not None:
        plt.savefig(savepath, facecolor="w")
    if fig is None and ax is None and close:
        plt.close()


def model_trio_lines(lr, e_vals, reg, reg_c, lw=1., k=1, skip0=True):
    # train valid uses 0, 1, first item uses 2, we should use 3 4 5
    n_plots = len(e_vals)-1
    EoS = 2/lr

    if skip0:
        bm_eig_lines = []
        bm_eig_lines.append((np.arange(n_plots) + 1, e_vals[1:, 0], 'line',
                             {'label': r"$λ_{max}$", 'color': dark_colours(0), 'linewidth': lw}))
        bm_eig_lines.append((np.arange(n_plots) + 1, EoS * np.ones(n_plots), 'line',
                             {'label': "EoS", "linestyle": "--", 'color': light_colours(0)}))

        neff_lines = []
        neff_lines.append((np.arange(n_plots) + 1, utils.compute_neff_k(e_vals[1:], reg, abs=True, k=k), 'line',
                           {'label': r"$SANE$", 'color': dark_colours(2), 'linewidth': lw}))
        neff_c_lines = []
        neff_c_lines.append((np.arange(n_plots) + 1, utils.compute_neff_const(e_vals[1:], reg_c, abs=True), 'line',
                             {'label': r"$N_{eff}$", 'color': dark_colours(3), 'linewidth': lw}))

    else:
        bm_eig_lines = []
        bm_eig_lines.append((np.arange(n_plots+1) , e_vals[:, 0], 'line',
                             {'label': r"$λ_{max}$", 'color': dark_colours(0), 'linewidth': lw}))
        bm_eig_lines.append((np.arange(n_plots+1), EoS * np.ones(n_plots+1), 'line',
                             {'label': "EoS", "linestyle": "--", 'color': light_colours(0)}))

        neff_lines = []
        neff_lines.append((np.arange(n_plots+1), utils.compute_neff_k(e_vals[:], reg, abs=True, k=k), 'line',
                           {'label': r"$SANE$", 'color': dark_colours(2), 'linewidth': lw}))
        neff_c_lines = []
        neff_c_lines.append((np.arange(n_plots+1), utils.compute_neff_const(e_vals[:], reg_c, abs=True), 'line',
                             {'label': r"$N_{eff}$", 'color': dark_colours(3), 'linewidth': lw}))

    return bm_eig_lines, neff_lines, neff_c_lines


def plot_model_trio(item_perm, item1, item2, trio_lines, xvs=[], fig=None, axs=None, labels=['valid', 'train', 'acc'], title0='acc'):

    if fig is None:
        assert axs is None
        fig, axs = plt.subplots(1, 3, figsize=(24, 6))

    n_plots = len(item_perm)
    bm_eig_lines, neff_lines, neff_c_lines = trio_lines
    item2_lines = [((np.arange(n_plots) + 1), item2, 'line', {'label': labels[2], 'color': dark_colours(2)})]
    item_perm_lines = [((np.arange(n_plots) + 1), item_perm, 'line', {'label': labels[0], 'color': dark_colours(0)})]

    if item1 is not None and len(item_perm) == len(item1):
    # plot loss
        item1_lines = [((np.arange(n_plots) + 1), item1, 'line', {'label': labels[1], 'color': dark_colours(1)})]
        lines1 = item_perm_lines + item1_lines
    else:
        lines1 = item_perm_lines

    plot_things_dual_axes(lines1, item2_lines, fig=fig, ax=axs[0], title=title0)
    plot_things_dual_axes(item_perm_lines, bm_eig_lines, fig=fig, ax=axs[1], title=f"sharp")
    plot_things_dual_axes(item_perm_lines, neff_lines, fig=fig, ax=axs[2], title=f"SANE")
    plot_things_dual_axes(item_perm_lines, neff_c_lines, fig=fig, ax=axs[3], title=f"neff_c")

    for xv in xvs:
        for ax in axs:
            ax.axvline(x=xv, c='k')

    # plt.show()
    return fig, axs


# importlib.reload(plotting)
def plot_model_trioW(grad_norm, eigsim, wsim, weights, grads, fig=None, axs=None, xvs=[]):

    if fig is None:
        assert axs is None
        fig, axs = plt.subplots(1, 3, figsize=(24, 6))

    n_plots = len(grads)

    sim_lines = [(np.arange(n_plots)+1, eigsim, 'line', {'label':'Hessian instability', 'color': dark_colours(1)})]
    sim_lines.append(((np.arange(n_plots)+1), wsim, 'line', {'label': 'Gradient instability', 'color': dark_colours(2)}))

    gradnorm_lines = [((np.arange(n_plots)+1, grad_norm, 'line', {'label': '|g|', 'color': dark_colours(0)}))]

    wdist = [np.linalg.norm(weights[t] - weights[0]) for t in np.arange(n_plots) + 1]
    rate_wdist = [wdist[t+1] - wdist[t] for t in range(len(wdist)-1)]
    # rate_wdist = [rate_wdist[0]] + rate_wdist
    wdist_lines = [(np.arange(n_plots)+1, wdist, 'line', {'label': '|w-w0|', 'color': dark_colours(4)})]
    rate_wdist_lines=[(np.arange(n_plots-1)+2, rate_wdist, 'line', {'label': 'd|w-w0|/dt', 'color': dark_colours(5)})]

    snr_window = 4
    snr = [utils.cos_sim(np.mean(grads[i:i+snr_window], axis=0), grads[i+snr_window], return_abs=True) for i in range(len(grads)-snr_window-1)]
    sim_og = [utils.cos_sim(weights[i+1]-weights[0], grads[i], return_abs=True) for i in range(len(grads))]
    snr_lines = [(np.arange(len(snr))+2+snr_window, snr, 'line', {'label': 'snr estimate', 'color': dark_colours(6)})]
    snr_lines.append((np.arange(n_plots)+1, sim_og, 'line', {'label': 'grad sim to w-w0', 'color': dark_colours(7)}))

    # sim_og_lines = [(np.arange(n_plots)+1, sim_og, 'line', {'label': 'grad sim to w-w0', 'color': light_colours(7)})]

    plot_things_dual_axes(gradnorm_lines, sim_lines, fig=fig, ax=axs[0], title=f"stability")
    plot_things_dual_axes(gradnorm_lines, wdist_lines, fig=fig, ax=axs[1], title=f"|w-w0|")
    plot_things_dual_axes(gradnorm_lines, rate_wdist_lines, fig=fig, ax=axs[2], title=f"d|w-w0|/dt")
    plot_things_dual_axes(gradnorm_lines, snr_lines, fig=fig, ax=axs[3], title=f"grad sim to w-w0")

    for xv in xvs:
        for ax in axs:
            ax.axvline(x=xv, c='k')

    return fig, axs


def plot_embed_line(inputs, interval=100, label_start=0, highlights = [], colour=None, label='_nolegend_', cmap='viridis', axs=None, fig=None, marker=None, s=5, line_alpha=1.0):
    if fig is None:
        assert axs is None
        fig, axs = plt.subplots(1, 1, figsize=(8, 6))

    x, y = inputs[:, 0], inputs[:, 1]
    indices = np.arange(len(x))
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    #     print(segments.shape)
    # fig, axs = plt.subplots(1, 1, figsize=(8, 4))

    # Create a continuous norm to map from data points to colors
    norm = plt.Normalize(indices.min(), indices.max())
    if colour is None:
        lc = LineCollection(segments, cmap=cmap, norm=norm, alpha=line_alpha)
    else:
        lc = LineCollection(segments, color=colour, norm=norm, alpha=line_alpha)
    # Set the values used for colormapping
    lc.set_array(indices)
    lc.set_linewidth(2)
    line = axs.add_collection(lc)
    if marker is not None:
        if colour is None:
            scatter = plt.scatter(inputs[:, 0], inputs[:, 1], c=np.arange(len(inputs)), cmap=cmap,marker=marker, label=label, s=s)
        else:
            scatter = plt.scatter(inputs[:, 0], inputs[:, 1], c=colour, marker=marker, label=label, s=s)

    # fig.colorbar(line, ax=axs)
    # axs.set_xlim(x.min(), x.max())
    # axs.set_ylim(y.min(), y.max())
    if interval > 0:
        n_int = int(x.shape[0] / interval)
        for i in range(n_int+1):
            try:
                tx, ty = x[i * interval], y[i * interval]
                plt.text(tx, ty, str(i * interval + label_start), c='k' if colour is None else colour)
            except:
                pass

    if len(highlights) > 0:
        for hl in highlights:
            tx, ty = x[hl], y[hl]
            plt.text(tx, ty, str(hl), c='r')
            scatter = plt.scatter(x[hl], y[hl], c='r', marker=marker, s=3*s)

    return fig, axs


def plot_embed_scatter(w, interval=100):
    assert w.shape[1] == 2  # assert is 2 dimensional

    cmap = matplotlib.cm.get_cmap('viridis')
    fig, axs = plt.subplots(1, 1, figsize=(8, 4))
    t = range(w.shape[0])
    scatter = plt.scatter(w[:, 0], w[:, 1], c=t, cmap=cmap)

    if interval > 0:
        n_int = int(w.shape[0] / interval)
        for i in range(n_int):
            x, y = w[i * interval]
            plt.text(x, y, str(i * interval), c='r')

    fig.colorbar(scatter, ax=axs)
    plt.show()


class hvlines():
    def __init__(self, lines, vertical=True, c='k', linestyle='--', linewidth=0.5):
        self.lines = lines
        self.vertical = vertical
        self.c = c
        self.linestyle = linestyle
        self.linewidth = linewidth

    def forward(self, ax, offset=0):
        plt.draw()

        for l in self.lines:
            ax.autoscale(False)
            if self.vertical:
                ax.axvline(x=l-offset, c=self.c, linestyle=self.linestyle, linewidth=self.linewidth)
            else:
                ax.axhline(y=l-offset, c=self.c, linestyle=self.linestyle, linewidth=self.linewidth)


from mpl_toolkits.mplot3d import Axes3D
import random

def vis_landscape(loss_fn, order=0, xmin=-10, xmax=10, ymin=-10, ymax=10, n_points=100, scale='linear', fig=None, ax=None):

    if fig is None:
        assert ax is None
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    xs = jnp.linspace(xmin, xmax, n_points)
    ys = jnp.linspace(ymin, ymax, n_points)
    X, Y = np.meshgrid(xs, ys)
    if order == 0:
        zs = np.array(loss_fn(np.ravel(X), np.ravel(Y)))
    elif order == 1:
        zs = np.array(jax.vmap(jax.grad(loss_fn, argnums=[0, 1]))(np.ravel(X), np.ravel(Y)))
        zs = np.linalg.norm(zs, axis=0)
    elif order == 2:
        zs = np.array(jax.vmap(jax.hessian(loss_fn, argnums=[0, 1]))(np.ravel(X), np.ravel(Y)))
        zs = np.max(zs, axis=(0, 1))
    else:
        raise NotImplementedError

    if scale == 'log':
        zs = np.log(zs)
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)

    ax.set_xlabel('Sharp Dir')
    ax.set_ylabel('Flat Dir')
    ax.set_zlabel('Loss - ' + scale)

    plt.show()

def vis_landscape_lgh(loss_fn, xmin=-2, xmax=2, ymin=-2, ymax=2, n_points=100, scale='linear', fig=None, ax=None):

    names = ['Loss', 'Grad', 'Hess']
    if fig is None:
        assert ax is None
        fig = plt.figure(figsize=plt.figaspect(0.33), layout='constrained')
        # fig = plt.figure()
        ax_l = fig.add_subplot(1, 3, 1, projection='3d')
        ax_g = fig.add_subplot(1, 3, 2, projection='3d')
        ax_h = fig.add_subplot(1, 3, 3, projection='3d')

    xs = jnp.linspace(xmin, xmax, n_points)
    ys = jnp.linspace(ymin, ymax, n_points)
    X, Y = np.meshgrid(xs, ys)
    zs_l = np.array(loss_fn(np.ravel(X), np.ravel(Y)))
    zs_g = np.array(jax.vmap(jax.grad(loss_fn, argnums=[0, 1]))(np.ravel(X), np.ravel(Y)))
    zs_g = np.linalg.norm(zs_g, axis=0)
    zs_h = np.array(jax.vmap(jax.hessian(loss_fn, argnums=[0, 1]))(np.ravel(X), np.ravel(Y)))
    zs_h = np.max(zs_h, axis=(0, 1))

    axs = [ax_l, ax_g, ax_h]
    zs = [zs_l, zs_g, zs_h]
    for i in range(3):
        z = np.log(zs[i]) if scale == 'log' else zs[i]
        ax = axs[i]
        Z = z.reshape(X.shape)
        ax.plot_surface(X, Y, Z)

        ax.set_xlabel('Sharp Dir')
        ax.set_ylabel('Flat Dir')
        # ax.set_zlabel(names[i] + ' - ' + scale)
        ax.set_title(names[i] + " - " + scale)
        ax.view_init(elev=60, azim=45, roll=15)

    plt.show()


def vis_landscape_lgh_q(loss_fn, xmin=-2, xmax=2, ymin=-2, ymax=2, yscale=False, n_points=20, scale='linear', do_g = True, do_h = True, fig=None, axs=None):

    names = ['Loss', 'Grad', 'Hess']
    if fig is None:
        assert axs is None
        fig = plt.figure(figsize=plt.figaspect(0.33), layout='constrained')
        # fig = plt.figure()
        ax_l = fig.add_subplot(1, 3, 1, projection='3d')
        ax_g = fig.add_subplot(1, 3, 2,)
        ax_h = fig.add_subplot(1, 3, 3,)
    else:
    	ax_l, ax_g, ax_h = axs

    xs = jnp.linspace(xmin, xmax, n_points)
    ys = jnp.linspace(ymin, ymax, n_points)
    X, Y = np.meshgrid(xs, ys)
    zs_l = np.array(loss_fn(np.ravel(X), np.ravel(Y)))

    zs_g = np.array(jax.vmap(jax.grad(loss_fn, argnums=[0, 1]))(np.ravel(X), np.ravel(Y)))
    u_g = zs_g[0].reshape(X.shape)
    v_g = zs_g[1].reshape(X.shape)
    w_g = np.zeros_like(X)
    zs_g = np.linalg.norm(zs_g, axis=0).reshape(X.shape)
    u_g /= zs_g
    v_g /= zs_g
    if yscale:
        sharp_scale = np.mean(np.abs(u_g))
        flat_scale = np.mean(np.abs(v_g))
        g_scale = sharp_scale / flat_scale
        v_g *= g_scale
        g_norm = np.linalg.norm(np.concatenate([u_g[np.newaxis, :], v_g[np.newaxis, :]], axis=0), axis=0)
        u_g /= g_norm
        v_g /= g_norm
    else:
        rangex = xmax-xmin
        rangey = ymax-ymin
        v_g *= rangex/rangey
        g_norm = np.linalg.norm(np.concatenate([u_g[np.newaxis, :], v_g[np.newaxis, :]], axis=0), axis=0)
        u_g /= g_norm
        v_g /= g_norm

    zs_h = np.array(jax.vmap(jax.hessian(loss_fn, argnums=[0, 1]))(np.ravel(X), np.ravel(Y)))
    s, v = jax.vmap(jnp.linalg.eigh)(zs_h.T)
    eigv_inds = np.argmax(s, axis=1)
    # eigvs = np.array([v[i, :, eigv_inds[i]] for i in range(len(s))])
    eigvs = v[:, :, 1].copy() # eigvs are sorted
    # print(v.shape, eigv_inds.shape, s.shape, eigvs.shape)
    u_h = eigvs[:, 0].reshape(X.shape)*-np.sign(X)
    v_h = eigvs[:, 1].reshape(X.shape)*-np.sign(X)
    if yscale:
        sharp_scale = np.mean(np.abs(u_h))
        flat_scale = np.mean(np.abs(v_h))
        h_scale = sharp_scale / flat_scale
        v_h *= h_scale
        h_norm = np.linalg.norm(np.concatenate([u_h[np.newaxis, :], v_h[np.newaxis, :]], axis=0), axis=0)
        u_h /= h_norm
        v_h /= h_norm
    else:
        rangex = xmax-xmin
        rangey = ymax-ymin
        v_h *= rangex/rangey
        h_norm = np.linalg.norm(np.concatenate([u_h[np.newaxis, :], v_h[np.newaxis, :]], axis=0), axis=0)
        u_h /= h_norm
        v_h /= h_norm


    zs_l = np.log(zs_l) if scale == 'log' else zs_l
    ax_l.plot_surface(X, Y, zs_l.reshape(X.shape), color='k', alpha=0.8)
    ax_l.set_xlabel('Sharp Dir')
    ax_l.set_ylabel('Flat Dir')

    if do_g:
        ax_g.quiver(X, Y, -u_g, -v_g)
        ax_g.set_xlabel('Sharp Dir')
        ax_g.set_ylabel('Flat Dir')
        ax_g.set_title(f"Grad")
        if yscale: print(f"Grad, scaled by {int(g_scale):d} for clarity")

    if do_h:
        ax_h.quiver(X, Y, u_h, v_h)
        ax_h.set_xlabel('Sharp Dir')
        ax_h.set_ylabel('Flat Dir')
        ax_h.set_title(f"Eigvecs")
        if yscale: print(f"Eigvecs, scaled by {int(h_scale):d} for clarity")

    if xmin < 0 and xmax > 0:
        if do_g:
            ax_g.axvline(x=0, c='k', linestyle='dashed', linewidth=0.2)
        if do_h:
            ax_h.axvline(x=0, c='k', linestyle='dashed', linewidth=0.2)
    if ymin < 0 and ymax > 0:
        if do_g:
            ax_g.axhline(y=0, c='k', linestyle='dashed', linewidth=0.2)
        if do_h:
            ax_h.axhline(y=0, c='k', linestyle='dashed', linewidth=0.2)


    axs = [ax_l, ax_g, ax_h]
    # plt.show()
    return fig, axs
