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
def vis_landscape(loss_fn, xmin=-2, xmax=2, ymin=-2, ymax=2, yscale=False, n_points=20, scale_fn=lambda x:x, do_g = True, do_h = True, fig=None, axs=None):

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
    loss_inputs = np.concatenate([np.ravel(X)[:, np.newaxis], np.ravel(Y)[:, np.newaxis]], axis=1)
    zs_l = np.array(loss_fn(loss_inputs))

    zs_g = np.array(jax.vmap(jax.grad(loss_fn))(loss_inputs))
    u_g = zs_g[:, 0].reshape(X.shape)
    v_g = zs_g[:, 1].reshape(X.shape)
    w_g = np.zeros_like(X)
    zs_g = np.linalg.norm(zs_g, axis=1).reshape(X.shape)
    u_g /= zs_g
    v_g /= zs_g

    if yscale:
        sharp_scale = np.mean(np.abs(u_g))
        flat_scale = np.mean(np.abs(v_g))
        g_scale = sharp_scale / flat_scale
        v_g *= g_scale
        g_norm = np.linalg.norm(np.concatenate([u_g[:, np.newaxis,], v_g[:, np.newaxis]], axis=1), axis=1)
        u_g /= g_norm
        v_g /= g_norm
    else:
        rangex = xmax-xmin
        rangey = ymax-ymin
        v_g *= rangex/rangey
        g_norm = np.linalg.norm(np.concatenate([u_g[:, np.newaxis,], v_g[:, np.newaxis,]], axis=1), axis=1)
        u_g /= g_norm
        v_g /= g_norm

    zs_h = np.array(jax.vmap(jax.hessian(loss_fn))(loss_inputs))
    s, v = jax.vmap(jnp.linalg.eigh)(zs_h)  # why was this transposed?
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
        h_norm = np.linalg.norm(np.concatenate([u_h[:, np.newaxis,], v_h[:, np.newaxis,]], axis=1), axis=1)
        u_h /= h_norm
        v_h /= h_norm

    else:
        rangex = xmax-xmin
        rangey = ymax-ymin
        v_h *= rangex/rangey
        h_norm = np.linalg.norm(np.concatenate([u_h[:, np.newaxis,], v_h[:, np.newaxis,]], axis=1), axis=1)
        u_h /= h_norm
        v_h /= h_norm

    # zs_l = np.log(zs_l) if scale == 'log' else zs_l
    zs_l = scale_fn(zs_l)
    ax_l.plot_surface(X, Y, zs_l.reshape(X.shape), color='k', alpha=0.7)
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
