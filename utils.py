import jax
import jax.numpy as jnp
import jax.random as rnd
import numpy as np
import optax
import torch
import os
from datetime import datetime, timedelta
import typing as tp
# import treex as tx
import jax.tree_util as tu
import jax.flatten_util as fu
import matplotlib.pyplot as plt

import inspect
from jax.scipy.signal import convolve
import pickle
import glob
import re
from scipy.stats import entropy
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from sklearn.neighbors import KernelDensity
import itertools
import math


#define jax types
Batch = tp.Mapping[str, np.ndarray]
# Model = tx.Sequential
Logs = tp.Dict[str, jnp.ndarray]

shade_colours = plt.get_cmap('Set3')
dark_colours = plt.get_cmap('tab10')
all_colours = plt.get_cmap('tab20')


def light_colours(i):
    return all_colours(2 * i + 1)


def get_now():
    now = datetime.now()
    dt_string = now.strftime("%y%m%d-%H%M")
    folders = glob.glob(f"traj/{dt_string}*")

    while len(folders) > 0:
        now = now + timedelta(minutes=1)
        dt_string = now.strftime("%y%m%d-%H%M")
        folders = glob.glob(f"traj/{dt_string}*")
    return dt_string


def play_beep(n=1):
    for i in range(n):
        if os.name == 'nt':
            import winsound
            frequency = 4000  # Set Frequency To 2500 Hertz\n"
            duration = 1000  # Set Duration To 1000 ms == 1 second\n",
            winsound.Beep(frequency, duration)
        elif os.name == 'posix':
            os.system("paplay /usr/share/sounds/gnome/default/alerts/glass.ogg")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def save_weights(state, path, verbose=False):
    save_thing(state.params, path)
    if verbose: print("weights saved to", path)


def save_opt(state, path, verbose=False):
    save_thing(state.opt_state, path)
    if verbose: print("weights saved to", path)


def save_thing(thing, path, verbose=False):
    with open(path, "wb") as file:
        p = pickle.Pickler(file)
        p.fast = True
        p.dump(thing)
    if verbose: print("thing saved to", path)


def load_thing(path, verbose=False):
    with open(path, "rb") as file:
        thing = pickle.load(file)
    if verbose: print("thing saved to", path)
    return thing


def count_weight_files(folder, freq=1):
    file_list = glob.glob(folder + "/w*.pkl")
    file_list.sort(key=lambda x: int(re.sub('\D', '', x.split("/")[-1])))
    final_ind = int(file_list[-1].split("/")[-1][1:-4])
    return final_ind


def get_all_weights(folder, freq=1):
    final_ind = count_weight_files(folder, freq)
    all_weights = []
    all_flat_weights = []

    for i in range(0, final_ind+1, freq):
        file = folder + "/w"+str(i)+".pkl"
        weights = load_thing(file)
        flat_weights, _ = fu.ravel_pytree(weights)
        all_weights.append(weights)
        all_flat_weights.append(flat_weights)

    return all_weights, np.array(all_flat_weights)


def get_thinned_w_grads(folder, freq=1):
    final_ind = count_weight_files(folder, freq)
    all_w_grads = []

    for i in range(freq, final_ind+1, freq):
        file = folder + "/w" + str(i-1) + ".pkl"
        next_file = folder + "/w" + str(i) + ".pkl"

        weights = load_thing(file)
        flat_weights, _ = fu.ravel_pytree(weights)

        next_weights = load_thing(next_file)
        flat_next_weights, _ = fu.ravel_pytree(next_weights)

        this_w_grad = flat_next_weights - flat_weights
        all_w_grads.append(this_w_grad)

    return np.array(all_w_grads)


def count_params(params):
    weights, _ = fu.ravel_pytree(params)
    return weights.shape[0]


def cos_sim(a, b, return_abs=True):
    if return_abs:
        return np.abs(np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b))
    else:
        return np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b)


def compare_cbs(b_list, a_str):
    for b in b_list:
        if b not in a_str:
            return False
    if "lrSchedule" in a_str:
        for b in b_list:
            if 'lrSchedule' in b:
                return True
        return False
    return True


def extract_number(f):
    s = re.findall("\d+$",f)
    return (int(s[0]) if s else -1,f)


def find_latest_exp(name, n_epochs, save_freq=1, cbs=None, verbose=False, unknown_lse=False,
                    resume=False, resume_n=0, traj_prefix='traj/'):
    time_str = name[:11]
    exp_details = name[11:]

    folder_list = glob.glob(f"{traj_prefix}*{exp_details}")
    folder_list.sort(reverse=True) # latest experiments first
    len_folder_skip = len(traj_prefix) # skipping the folder string

    last_save_epoch = n_epochs
    while (last_save_epoch % save_freq != 0): # make sure this one is actually saved
        last_save_epoch -= 1

    if verbose: print(folder_list)

    # check folders
    for folder in folder_list:
        # Reject if callbacks do not match
        if cbs is not None:
            # If using callbacks file
            if os.path.exists(folder+"/callbacks.pkl"):
                cb_name_load = load_thing(folder+"/callbacks.pkl")

            # If not using callbacks file
            else:
                len_skip = 5+12+len(exp_details)+1 - 1
                if verbose: print("no cb file", folder)
                # reject if folder has no CBs
                if len(folder) < len_skip + 1:
                    continue
                cb_name_load = folder[len_skip:] # traj/ + date + exp_name = 5+12+len(exp_name) + 1

            # reject if CBs from list does not exist in the string
            cb_fit = compare_cbs(cbs, cb_name_load)
            if not cb_fit:
                if verbose: print(cbs, cb_name_load)
                continue

        # If not rejected by CBs, check last weight file exists and return
        if not unknown_lse:
            if os.path.exists(folder + "/w" + str(last_save_epoch) + ".pkl"):
                return folder[len_folder_skip:], last_save_epoch
            else:
                if resume:
                    list_of_files = glob.glob(folder+"/w*")  # * means all if need specific format then *.csv
                    list_of_files = [x.split("/")[1] for x in list_of_files]
                    latest_file = max(list_of_files, key=extract_number)
                    return folder[len_folder_skip:], int(latest_file[1:-4])
                else:
                    if verbose: print("no lse found")
        else:
            # print("unknown lse")
            es_files = glob.glob(folder+"/early_stop*")
            if verbose: print("unknown lse", es_files)
            if len(es_files) > 0:
                # print(es_files[0])
                last_save_epoch = int(es_files[0].split("/")[-1][10:-4])
                return folder[len_folder_skip:], last_save_epoch
            else:
                if resume:
                    print("I'm here!", folder)
                    list_of_files = glob.glob(folder+"/w*")  # * means all if need specific format then *.csv
                    list_of_lses = [int(x.split("/")[-1][1:-4]) for x in list_of_files]
                    # latest_file = max(list_of_files, key=extract_number)
                    # print(list_of_lses)
                    if max(list_of_lses)>resume_n:
                        return folder[len_folder_skip:], max(list_of_lses)
                    else:
                        if verbose: print("training ended too early", max(list_of_lses))
                else:
                    if verbose: print("no lse/resume found")

    raise FileNotFoundError


def find_latest_exp_no_epoch(name, cbs=None, verbose=False, max_eps=0,
                    resume=False, resume_n=0, traj_prefix='traj/'):
    # save code as above, but assuming always no LSE, and remove n_epochs and save_freq as inputs
    time_str = name[:11]
    exp_details = name[11:]

    folder_list = glob.glob(f"{traj_prefix}*{exp_details}")
    folder_list.sort(reverse=True) # latest experiments first
    len_folder_skip = len(traj_prefix) # skipping the folder string

    if verbose: print(folder_list)

    # check folders
    for folder in folder_list:
        # Reject if callbacks do not match
        if cbs is not None:
            # If using callbacks file
            if os.path.exists(folder+"/callbacks.pkl"):
                cb_name_load = load_thing(folder+"/callbacks.pkl")

            # If not using callbacks file
            else:
                len_skip = 5+12+len(exp_details)+1 - 1
                if verbose: print("no cb file", folder)
                # reject if folder has no CBs
                if len(folder) < len_skip + 1:
                    continue
                cb_name_load = folder[len_skip:] # traj/ + date + exp_name = 5+12+len(exp_name) + 1

            # reject if CBs from list does not exist in the string
            cb_fit = compare_cbs(cbs, cb_name_load)
            if not cb_fit:
                if verbose: print(cbs, cb_name_load)
                continue

        # If not rejected by CBs, check last weight file exists and return
        # print("unknown lse")
        es_files = glob.glob(folder+"/early_stop*")

        if len(es_files) > 0:
            # print(es_files[0])
            last_save_epoch = int(es_files[0].split("/")[-1][10:-4])
            return folder[len_folder_skip:], last_save_epoch
        else:
            nc_files = glob.glob(folder + "/no_converge*")
            if len(nc_files) > 0:
                last_save_epoch = int(nc_files[0].split("/")[-1][11:-4])
                if last_save_epoch >= max_eps:
                    return folder[len_folder_skip:], last_save_epoch
                else:
                    if resume:
                        print("I'm here!", folder)
                        list_of_files = glob.glob(folder+"/w*")  # * means all if need specific format then *.csv
                        list_of_lses = [int(x.split("/")[-1][1:-4]) for x in list_of_files]
                        # latest_file = max(list_of_files, key=extract_number)
                        # print(list_of_lses)
                        if max(list_of_lses)>resume_n:
                            return folder[len_folder_skip:], max(list_of_lses)
                        else:
                            if verbose: print("training ended too early", max(list_of_lses))
                    else:
                        if verbose: print("no lse/resume found")

            else:
                continue

    raise FileNotFoundError



def thin_pickle(path, n_pl=None, pf=None, has0=True):

    thing = load_thing(path)

    if n_pl is None:
        thin_freq = pf
        if has0:
            num_pl = math.ceil((len(thing)-1)/thin_freq)
        else:
            num_pl = math.ceil((len(thing))/thin_freq)
        new_path = path[:-4] + "_thin" + str(num_pl) + ".pkl"

        try:
            if has0:
                thin_thing = [thing[i] for i in range(0, len(thing), thin_freq)] # from 0 onwards
            else:
                assert len(thing) % thin_freq == 0
                thin_thing = [thing[i] for i in range((len(thing)-1)%thin_freq, len(thing), thin_freq)] # from plot_freq-1 onwards
        except AssertionError:
            print(n_pl, thin_freq, len(thing), len(thing)%thin_freq)
            raise(AssertionError)
    else:
        thin_freq = int(len(thing) / n_pl)
        new_path = path[:-4] + "_thin" + str(n_pl) + ".pkl"

        try:
            if has0:
                if thin_freq > 1 and n_pl > 1 and pf is None:
                    assert len(thing) % thin_freq == 1
                thin_thing = [thing[i] for i in range(0, len(thing), thin_freq)]  # from 0 onwards
            else:
                assert len(thing) % thin_freq == 0
                thin_thing = [thing[i] for i in range((len(thing) - 1) % thin_freq, len(thing), thin_freq)]  # from plot_freq-1 onwards
        except AssertionError:
            print(n_pl, thin_freq, len(thing), len(thing) % thin_freq)
            raise (AssertionError)

    save_thing(thin_thing, new_path)


def get_callback_name_str(callbacks):
    cb_name = ""
    for i, cb in enumerate(callbacks):
        if i > 0:
            cb_name += "_"
        cb_name += cb.name
    return cb_name


def get_callback_name_list(callbacks):
    cb_name = []
    for i, cb in enumerate(callbacks):
        cb_name.append(cb.name)
    return cb_name


def depth_vlines(model, wb=False):
    # reparamCB
    weights, _ = tu.tree_flatten(model)
    shapes = []
    tmp = 0
    for i, w in enumerate(weights):
        if not wb:
            if (i+1) % 2 == 0:
                shapes.append(tmp + np.prod(w.shape))
                tmp = 0
            else:
                tmp += np.prod(w.shape)
        else:
            shapes.append(np.prod(w.shape))
    return np.cumsum(shapes)


def compute_neff_entropy(e_vals, reg=2., abs=True):
    if abs:
        e_vals = np.abs(e_vals)
    len_t = len(e_vals)
    line = []
    for t in range(len_t):
        # alpha = e_vals[t].max() * reg
        line.append(entropy(e_vals[t], base=2))
    return np.array(line)


def compute_neff(e_vals, reg=0.5, abs=True):
    if abs:
        e_vals = np.abs(e_vals)
    len_t = len(e_vals)
    line = []
    for t in range(len_t):
        alpha = e_vals[t].max() * reg
        line.append(np.sum(e_vals[t] / (e_vals[t] + alpha)))
    return np.array(line)


def compute_neff_k(e_vals, reg=0.5, abs=True, k=1):
    if abs:
        e_vals = np.abs(e_vals)
    len_t = len(e_vals)
    line = []
    for t in range(len_t):
        sorted_evals = np.sort(e_vals[t])[::-1]
        alpha = sorted_evals[k-1] * reg
        line.append(np.sum(e_vals[t] / (e_vals[t] + alpha)))
    return np.array(line)


def compute_neff_k_neg0(e_vals, reg=0.5, k=1):
    e_vals = np.clip(e_vals, 0, np.nanmax(e_vals))
    len_t = len(e_vals)
    line = []
    for t in range(len_t):
        sorted_evals = np.sort(e_vals[t])[::-1]
        alpha = sorted_evals[k-1] * reg
        line.append(np.sum(e_vals[t] / (e_vals[t] + alpha)))
    return np.array(line)


def compute_neff_hessvar(e_vals, reg=0.5, abs=True):
    len_t = len(e_vals)
    line = []
    for t in range(len_t):
        alpha = np.abs(-2*e_vals[t].min()*reg)
        if abs:
            line.append(np.sum(np.abs(e_vals[t]) / (np.abs(e_vals[t]) + alpha)))
        else:
            line.append(np.sum(e_vals[t] / (e_vals[t] + alpha)))
    return np.array(line)


def compute_neff_const(e_vals, reg=0.5, abs=True):
    if abs: e_vals = np.abs(e_vals)
    len_t = len(e_vals)
    line = []
    for t in range(len_t):
        alpha = reg
        line.append(np.sum(e_vals[t] / (e_vals[t] + alpha)))
    return np.array(line)


def compute_neff_const_neg0(e_vals, reg=0.5):
    e_vals = np.clip(e_vals, 0, np.nanmax(e_vals))
    len_t = len(e_vals)
    line = []
    for t in range(len_t):
        alpha = reg
        line.append(np.sum(e_vals[t] / (e_vals[t] + alpha)))
    return np.array(line)


def compute_neff_k_pow(e_vals, reg=0.5, abs=True, k=1, pow=0.5):
    if abs:
        e_vals = np.abs(e_vals)
    len_t = len(e_vals)
    line = []
    for t in range(len_t):
        sorted_evals = np.sort(e_vals[t])[::-1]
        alpha = sorted_evals[k-1]**pow * reg
        line.append(np.sum(e_vals[t] / (e_vals[t] + alpha)))
    return np.array(line)


def dinh_reparam(weights, scheme, nnidx):
    new_weights = np.array(weights).copy()
    assert len(nnidx[0]) == 3  # has weights and biases
    assert np.cumprod(scheme)[-1] == 1
    for i in range(len(scheme)):
        if i == 0:
            new_weights[nnidx[i][0]:nnidx[i][-1]] *= scheme[i]  # weights and biases
            K = scheme[i]  # factor of this layer increase
        else:
            M = scheme[i]
            K_new = M * K  # T = m*k
            new_weights[nnidx[i][1]:nnidx[i][2]] *= M  # just weights
            new_weights[nnidx[i][0]:nnidx[i][1]] *= K_new  # just biases
            K = K_new

    if len(scheme) < len(nnidx):
        new_weights[nnidx[i][1]:nnidx[i][2]] /= K_new
    else:
        raise ValueError("Weights reparam scheme must not include the output layer ")

    return new_weights


def dk_cos(ma, mb, verbose=False):
    # verified correct and works for multi dimensions of a and b. The first axis of the inputs denote the axis of the whole space, while the second is the differing axis.
    # Could consider changing in future
    ma_norm = ma / np.linalg.norm(ma, axis=0)[np.newaxis, :]
    mb_norm = mb / np.linalg.norm(mb, axis=0)[np.newaxis, :]
    assert ma.shape[0] == mb.shape[0]
    dim_all = ma.shape[0]
    dim_a = ma.shape[1]
    dim_b = mb.shape[1]
    M = ma_norm.T @ mb_norm
    u, s, vh = jnp.linalg.svd(M)
    s = np.clip(s, -1, 1)
    thetas = jnp.arccos(s) # same dimension
    if verbose: print(thetas)
    return jnp.cos((jnp.sqrt(jnp.sum(thetas**2)) / jnp.sqrt((min(dim_a, dim_b) * (np.pi/2)**2))) * np.pi/2)


def dk_cos_lin(ma, mb, verbose=False):
    # verified correct and works for multi dimensions of a and b. The first axis of the inputs denote the axis of the whole space, while the second is the differing axis.
    # Could consider changing in future
    ma_norm = ma / np.linalg.norm(ma, axis=0)[np.newaxis, :]
    mb_norm = mb / np.linalg.norm(mb, axis=0)[np.newaxis, :]
    assert ma.shape[0] == mb.shape[0]
    dim_all = ma.shape[0]
    dim_a = ma.shape[1]
    dim_b = mb.shape[1]
    M = ma_norm.T @ mb_norm
    u, s, vh = jnp.linalg.svd(M)
    # if verbose: print("s unclipped", s) # should have no need for clipping now
    s = np.clip(s, -1, 1)
    thetas = jnp.arccos(s) # same dimension
    if verbose: print(s, thetas)
    return jnp.cos((jnp.sum(thetas**2) / ((min(dim_a, dim_b) * (np.pi/2)**2))) * np.pi/2)


def compute_entropy(e_vals, rounding=0, method='kde'):
    len_t = len(e_vals)
    line = []
    for t in range(len_t):
        if method == 'kde':
            data = e_vals[t][:, np.newaxis]
            if np.isnan(data).all():
                line.append(np.nan)
            else:
                kde = KernelDensity(kernel='gaussian', bandwidth=10 ** (-rounding)).fit(data)
                log_p = kde.score_samples(data)  # returns log(p) of data sample
                p = np.exp(log_p)  # estimate p of data sample
                entropy_estimate = -np.sum(p * log_p)  # evaluate entropy
                line.append(entropy_estimate)

        else:
            # box histogram method
            e_vals_tmp = np.around(e_vals[t].copy(), decimals=rounding)
            uniques, counts = np.unique(e_vals_tmp, return_counts=True)
            line.append(entropy(counts, base=2))

    return np.array(line)


def tmi_cond_call(a):
    def call(tbm, ind):
        tbm = jax.lax.cond(a[tbm[0]]>a[tbm[1]], lambda x: jnp.array([x[0] + 1, x[1]]), lambda x: jnp.array([x[0], x[1]-1]), tbm)
        return tbm, ind
    return call

def get_top_mag_inds_jax(a, n=1):
    # get inclusive indices of top magnitude items from a sorted np list
    abs_a = jnp.abs(a)

    tbm = jnp.array([0, len(a)-1])
    tmi_cond_cl = tmi_cond_call(abs_a)
    # print(tmi_cond_cl(tbm))
    tbm, _ = jax.lax.scan(tmi_cond_cl, tbm, jnp.arange(n))
    # print(tbm)
    return int(tbm[0]-1), int(tbm[1]+1)


def get_top_mag_inds(a, n=1):

    top = 0
    bottom = len(a) - 1
    for i in range(n):
        if np.abs(a[top]) > np.abs(a[bottom]):
            top += 1
        else:
            bottom -= 1
    return top-1, bottom+1

    # top + bottom = len(a) - 1
    # top + len(a) - bottom = n

    # two way binary search would be best but im pressed for time


def rand_radamacher(shape, rng_key):
    init_vec = jax.random.uniform(rng_key, shape=shape, dtype=np.float32)
    return ((init_vec<0.5)*2. - 1.)


def mat_mse(mat, verbose=False):
    # mat[i, j] is the sim between source_i and target_j
    # we want orient to get the best target_j for each source_i
    best_mse = np.inf

    n = len(mat) # number of sources
    m = len(mat[0]) # number of targets

    skip_inds = []
    skip_oris = []
    maxs = np.max(mat, axis=1) # for each source, which target is best

    for i in range(n):
        if maxs[i] > np.sqrt(2) / 2:
            skip_inds.append(i)
            skip_oris.append(np.argmax(mat[i, :]))

    list_nums = [i for i in range(n) if i not in skip_inds] # list of remainders over all sources
    iter_nums = [i for i in range(m) if i not in skip_oris] # list iter nums over all targets

    if verbose:
        # debug
        print("max", maxs, "remainder inds", list_nums)
        for i in range(len(skip_inds)):
            print("s", skip_inds[i], "t", skip_oris[i], "sim", maxs[skip_inds[i]])

    if len(list_nums) > 0:
        nln = len(list_nums)
        assert nln == n - len(skip_inds)
        nin = len(iter_nums)
        assert nin == m - len(skip_oris)

        mat_red = mat.copy()
        mat_red = mat_red[list(list_nums), :]
        mat_red = mat_red[:, list(iter_nums)]
        #         print(mat_red.shape)

        list_perms = itertools.permutations(range(nin), nln)
        for lp in list_perms:
            # if np.sum()
            mat_tmp = mat_red.copy()
            mat_tmp = mat_tmp[:, lp]
            mse = np.sum((np.diag(mat_tmp) - np.ones(nln)) ** 2) / n

            if mse < best_mse:
                red_orient = lp
                best_mse = mse

        orient = np.zeros(n, dtype=np.int32)
        rd_count = 0
        skip_count = 0
        for i in range(n):
            if i in skip_inds:
                orient[i] = skip_oris[skip_count]
                skip_count += 1
            else:
                orient[i] = iter_nums[red_orient[rd_count]]
                rd_count += 1
        orient = list(orient)
    else:
        orient = skip_oris
    mat_tmp = mat.copy()
    mat_tmp = mat_tmp[:, orient]
    best_mse = np.sum((np.diag(mat_tmp) - np.ones(n)) ** 2) / n
    if verbose: print("tmp", mat_tmp)
    return best_mse, orient, skip_inds


def mat_mse_copy(mat, verbose=False): # working version
    # mat[i, j] is the sim between source_i and target_j
    # we want orient to get the best target_j for each source_i
    best_mse = np.inf

    n = len(mat) # number of sources
    m = len(mat[0]) # number of targets

    skip_inds = []
    skip_oris = []
    maxs = np.max(mat, axis=0) # for each source, which target is best

    for i in range(n):
        if maxs[i] > np.sqrt(2) / 2:
            skip_inds.append(i)
            skip_oris.append(np.argmax(mat[:, i]))

    list_nums = [i for i in range(m) if i not in skip_oris] # list nums over all targets

    if verbose:
        # debug
        print(maxs, list_nums)
        for i in range(len(skip_inds)):
            print(skip_inds[i], skip_oris[i], maxs[skip_inds[i]])

    if len(list_nums) > 0:
        nin = len(list_nums)
        mat_red = mat.copy()
        mat_red = mat_red[list(list_nums), :]
        mat_red = mat_red[:, list(list_nums)]
        #         print(mat_red.shape)

        list_perms = itertools.permutations(range(nin), nin)
        for lp in list_perms:
            mat_tmp = mat_red.copy()
            mat_tmp = mat_tmp[:, lp]
            mse = np.sum((mat_tmp.flatten() - np.eye(nin).flatten()) ** 2) / n ** 2

            if mse < best_mse:
                red_orient = lp
                best_mse = mse

        orient = np.zeros(n, dtype=np.int32)
        rd_count = 0
        skip_count = 0
        for i in range(n):
            if i in skip_inds:
                orient[i] = skip_oris[skip_count]
                skip_count += 1
            else:
                orient[i] = list_nums[red_orient[rd_count]]
                rd_count += 1
        orient = list(orient)
    else:
        orient = skip_oris
    mat_tmp = mat.copy()
    mat_tmp = mat_tmp[:, orient]
    best_mse = np.sum((mat_tmp.flatten() - np.eye(n).flatten()) ** 2) / n ** 2

    return best_mse, orient.astype(np.int32), skip_inds


def moving_average(x, w=1):
    conv = np.convolve(x, np.ones(w), 'valid')
    return  conv / w

def normalise(x):
    return (x-np.min(x))/(np.max(x)-np.min(x))


def get_peaks(loss, eigvals=None, prominence=0.1, reverse=False):
    loss = np.array(loss)
    peaks = np.array(find_peaks(loss, prominence=prominence)[0])
    if not reverse:
        if eigvals is not None:
            for i in range(len(peaks)):
                flag = True
                while flag:
                    try:
                        if eigvals[peaks[i] + 1, 0] > eigvals[peaks[i], 0]:
                            peaks[i] += 1
                        else:
                            flag = False
                    except IndexError:
                        flag = False
        return peaks

    else:
        assert eigvals is not None
        rev_peaks = []
        for i in range(len(peaks)-1):
            rev_peaks.append(np.argmin(eigvals[peaks[i]:peaks[i+1]])+peaks[i])
        return rev_peaks


def get_peaks_eigvals(eigvals, lr=None, prominence=0.1, reverse=False):
    eigvals = np.array(eigvals)
    if not reverse:
        if lr is not None:
            peaks = np.array(find_peaks(eigvals, prominence=prominence, height=2/lr)[0])
        else:
            peaks = np.array(find_peaks(eigvals, prominence=prominence)[0])
        return peaks
    else:
        if lr is not None:
            peaks = np.array(find_peaks(-eigvals, prominence=prominence, height=-2/lr)[0])
        else:
            peaks = np.array(find_peaks(-eigvals, prominence=prominence)[0])
        # rev_peaks = []
        # for i in range(len(peaks)-1):
        #     rev_peaks.append(np.argmin(eigvals[peaks[i]:peaks[i+1]])+peaks[i])
        return peaks


def signif(x, p=3):
    x = np.asarray(x)
    x_positive = np.where(np.isfinite(x) & (x != 0), np.abs(x), 10**(p-1))
    mags = 10 ** (p - 1 - np.floor(np.log10(x_positive)))
    return np.round(x * mags) / mags


def reject_outliers(data, m=100., n_ma=0.):
    data = np.array(data)
    d = np.abs(data - np.median(data))
    mdev = np.median(d)
    s = d / (mdev if mdev else 1.)
    #     return data[s < m]
    data[s > m] = np.nan
    if len(data[s < m]) > 0 and n_ma > 0:
        return np.convolve(data, np.ones(n_ma) / n_ma, mode='valid')
    else:
        return data


def max_dk(a, b, n):
    return max(dk_cos(a[:, :n], b,), dk_cos(a, b[:, :n]))


def cifar5k_lr_fix(counter):
    lrs = [0.0002, 0.00025, 0.0003, 0.00035, 0.0004, 0.00045, 0.0005, 0.000625, 0.00075, 0.000875, 0.001, 0.00125,
           0.0015]
    sws = [10000, 10000, 10000, 10000, 10000, 10000, 1000, 1000, 1000, 100, 100, 10, 100]
    neps = [250000, 250000, 250000, 250000, 250000, 250000, 100000, 100000, 100000, 20000, 20000, 20000, 20000]
    for i in range(20):
        lrs.append(signif(0.002 * 1.6 ** i, 3))
        sws.append(10)
        neps.append(20000)

    if isinstance(counter, int):
        return lrs[counter], sws[counter], neps[counter]
    else:
        if counter in lrs:
            ind = lrs.index(counter)
            return lrs[ind], sws[ind], neps[ind]
        else:
            raise KeyError

def optim_to_option(optim):
    if optim == 'asam':
        return 'sam'
    elif optim == 'sam':
        return 'sam'
    elif optim == 'looksam':
        return 'sam'

