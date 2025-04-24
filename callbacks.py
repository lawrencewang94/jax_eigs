import copy

import jax
import jax.numpy as jnp
import numpy as np
import pickle
import optax
import os
import math
import queue

# import refactor.utils as utils
# import refactor.spectral as spectral
# import refactor.training as training
import utils
import spectral
import training

import typing as tp
import jax.tree_util as tu
import jax.flatten_util as fu
from scipy.optimize import minimize as glob_opt
# from training import test_step


import importlib
importlib.reload(spectral)
importlib.reload(utils)

pi = jnp.pi
exp = jnp.exp

Batch = tp.Mapping[str, np.ndarray]
# Model = tx.Sequential
Logs = tp.Dict[str, jnp.ndarray]


class Callback:
    def __init__(self, save_freq=1, save_pref="traj/", verbose=False):
        self.save_freq = save_freq
        self.save_pref = save_pref
        self.save_path = ""
        self.verbose = verbose
        self.name = "callback_" + str(save_freq)

    def init(self, **kwargs):
        self.save_path = self.save_pref + kwargs['exp_name']
        if self.save_freq > 0 and not os.path.exists(self.save_path):
            os.mkdir(self.save_path)

    def forward(self, **kwargs):
        pass

    def get_name(self):
        return self.name

    def save(self):
        pass


class saveWeightsCB(Callback):
    def __init__(self, save_freq, save_pref="traj/", grad=True, verbose=False):
        super().__init__(save_freq=save_freq, save_pref=save_pref, verbose=verbose)
        self.name = "saveW_" + str(save_freq)
        self.save_grad = grad

    def forward(self, **kwargs):
        epoch = kwargs['epoch']
        # if epoch % self.save_freq == 0 or epoch % self.save_freq == 1 or (epoch + 1) % self.save_freq == 0:  # removed modulus = 1, define w_grad[i] = w[i] - w[i-1]
        if epoch % self.save_freq == 0 or (self.save_grad==True and (epoch + 1) % self.save_freq == 0):

            state = kwargs['state']
            utils.save_weights(state, self.save_path + "/w" + str(epoch) + ".pkl")


class hessianCB(Callback):
    def __init__(self, loss_fn, batch, save_freq, bn=True, save_pref="traj/", verbose=False):
        super().__init__(save_freq=save_freq, save_pref=save_pref, verbose=verbose)
        self.name = "saveHess_" + str(save_freq)
        self.batch = batch
        self.bn = bn
        self.loss_fn = loss_fn
        self.loss_wrap = None
        self.hessians = []
        self.last_hess = None
        self.num_params = 0

    def forward(self, **kwargs):
        epoch = kwargs['epoch']
        if epoch % self.save_freq == 0:
            state = kwargs['state']
            if self.loss_wrap is None:
                self.loss_wrap = spectral.get_loss_wrap(state, self.loss_fn, bn=self.bn)  # get fn that computes loss from (model, data)

            try:
                train = kwargs['train']
            except KeyError:
                train = True

            get_loss = lambda w: self.loss_wrap(w, self.batch, train=train)
            self.last_hess = jax.hessian(get_loss)(state.params)
            self.hessians.append(self.last_hess)
            self.num_params = utils.count_params(state.params)

    def save(self):
        utils.save_thing(np.array(self.hessians), self.save_path + f"/hessians.pkl")
        print("hessians saved!")


class hvpCB(Callback):
    def __init__(self, loss_fn, batches, save_freq, state=None, hess_bs=0, bn=True, save_pref="traj/", verbose=False):
        super().__init__(save_freq=save_freq, save_pref=save_pref, verbose=verbose)
        self.name = "hvp_"+str(save_freq)
        self.batches = batches if len(batches) == 2 else batches[0]
        self.bn = bn
        self.loss_fn = loss_fn
        self.hess_bs = hess_bs

        self.loss_wrap = None
        self.num_params = 0
        self.hvp = None

        if state is not None:
            self.build_hvp(state)

    def build_hvp(self, state):
        self.loss_wrap = spectral.get_loss_wrap(state, self.loss_fn, bn=self.bn)  # get fn that computes loss from (model, data)

        # print(self.hess_bs, len(self.batches[0]))
        hvp_fn, self.model_structure, self.num_params = spectral.get_hvp_fn(self.loss_wrap, state, self.batches, self.hess_bs)  # get fn that computes hvp from (model, v)
        # if self.hess_bs == 0 or self.hess_bs>= len(self.batches[0]): self.hvp = jax.jit(lambda w, v: hvp_fn(w, v))  # get fn that computes hvp from (w, v)
        # else: self.hvp = jax.jit(lambda w, v: hvp_fn(w, v))  # get fn that computes hvp from (w, v)
        self.hvp = jax.jit(lambda w, v: hvp_fn(w, v)) # jit for both, yay!

    def forward(self, **kwargs):
        if self.loss_wrap is None:
            state = kwargs['state']
            self.build_hvp(state)
            # hvp_fn, self.model_structure, self.num_params = spectral.get_hvp_fn(self.loss_wrap, model, self.batches)  # get fn that computes hvp from (model, v)
            # self.hvp = jax.jit(lambda w, v: hvp_fn(w, v))  # get fn that computes hvp from (w, v)


class spectrumCB(Callback):
    def __init__(self, n_eigs=0, n_evecs=0, loss_fn=None, seed=0, hessianCB=None, hvpCB=None,
                 save_freq=1, save_pref="traj/", verbose=False, save_name_prefix=""):
        super().__init__(save_freq=save_freq, save_pref=save_pref, verbose=verbose)

        assert (hessianCB is not None) or (hvpCB is not None)
        self.full = hessianCB is not None
        self.hessianCB = hessianCB
        self.hvpCB = hvpCB

        self.n_eigs = -1 if self.full else n_eigs
        self.n_evecs = min(n_evecs, n_eigs) if not self.full else n_eigs
        self.loss_fn = loss_fn
        self.seed = seed
        self.rng = jax.random.PRNGKey(self.seed)

        self.all_eigvals = []
        self.all_eigvecs = []
        self.all_eigvec_eigvals = []
        self.last_eigval = None
        self.last_eigvec = None
        self.last_eigvec_eigval = None
        self.last_tridiag = None
        name_method = "spectrumHess" if self.full else "spectrumHvp"
        self.name = f"{name_method}_{self.n_eigs:d}_{self.n_evecs:d}_{save_freq:d}_{save_name_prefix}"
        self.save_name_prefix = save_name_prefix

    def forward(self, **kwargs):
        epoch = kwargs['epoch']
        if epoch % self.save_freq == 0:
            if self.full:
                if self.n_eigs == -1:
                    self.n_eigs = self.hessianCB.num_params
                hessian = self.hessianCB.last_hess
                eigvals, eigvecs = np.linalg.eig(hessian)
                eigvecs = eigvecs.T # so first index is the eigvalue index
                e_vec_evals = eigvecs.copy()
            else:
                self.num_params = self.hvpCB.num_params
                # weights, _ = fu.ravel_pytree(kwargs['model'])
                # flat_weights, _ = fu.ravel_pytree(kwargs['model'].parameters()) # edited for TRANSFORMER
                self.rng, split_rng = jax.random.split(self.rng)

                state = kwargs['state']
                #hvp is already jitted
                hvp_cl = lambda v: self.hvpCB.hvp(state.params, v)

                tridiag, lanc_vecs = spectral.lanczos_alg(hvp_cl, self.num_params, self.n_eigs, split_rng)  # perform lanczos with hvp_cl
                e_vals, tri_evecs, eigval_weights = spectral.tridiag_to_eigv([tridiag])  # compute evals
                # print("spectrum debug", tri_evecs[0].T.shape, lanc_vecs.shape) # ord X ord, ord X dim
                # e_vecs = tri_evecs[0].T[:, :self.n_evecs] @ lanc_vecs[:self.n_evecs, :]  # compute evecs; also verified correct; dim reduction: ord X red, red X dim

                top_ind, bottom_ind =  utils.get_top_mag_inds(e_vals[0], n=self.n_evecs)
                e_vecs_top = jnp.matmul(tri_evecs[0].T[:top_ind + 1, :], lanc_vecs)  # compute evecs; also verified correct; dim reduction: red X ord, ord X dim = red X dim
                e_vecs_btm = jnp.matmul(tri_evecs[0].T[bottom_ind:, :], lanc_vecs)  # compute evecs; also verified correct; dim reduction: red X ord, ord X dim = red X dim

                # print("spectral debug", e_vals.shape, e_vecs.shape, e_vals)
                eigvals = np.flip(e_vals[0], axis=0)
                e_vec_evals = np.flip(np.concatenate([e_vals[0][:top_ind+1], e_vals[0][bottom_ind:]], axis=0), axis=0)
                # eigvecs = np.flip(e_vecs, axis=0)
                eigvecs = np.flip(np.concatenate([e_vecs_top, e_vecs_btm], axis=0), axis=0)
                # print("spectral debug2", eigvals.shape, eigvecs.shape, eigvals)

            self.all_eigvals.append(eigvals)
            self.all_eigvecs.append(eigvecs)
            self.all_eigvec_eigvals.append(e_vec_evals)
            self.last_eigval = eigvals.copy()
            self.last_eigvec = eigvecs.copy()
            self.last_eigvec_eigval = e_vec_evals.copy()
            if not self.full: self.last_tridiag = tridiag.copy()
            if epoch == 0:
                print("init top eigenvalue", self.all_eigvals[0][0].real)
            if self.verbose: print("sharpness", self.last_eigval[0])

    def save(self):
        print("final top eigenvalue", self.all_eigvals[-1][0].real)
        utils.save_thing(np.array(self.all_eigvals), self.save_path+"/"+self.save_name_prefix+"eigvals.pkl")
        if self.n_evecs > 0:
            print(np.array(self.all_eigvecs).shape)
            utils.save_thing(np.array(self.all_eigvecs), self.save_path+"/"+self.save_name_prefix+"eigvecs.pkl")
            utils.save_thing(np.array(self.all_eigvec_eigvals), self.save_path+"/"+self.save_name_prefix+"eigvec_eigvals.pkl")
        print("Eigenvectors and eigenvalues saved!")


class thinCB(Callback):
    def __init__(self, n_pl=None, thin_freq=None, save_freq=0, save_pref="traj/", verbose=False):
        super().__init__(save_freq=save_freq, save_pref=save_pref, verbose=verbose)
        self.n_pl = n_pl
        self.thin_freq = thin_freq
        self.name = "thin_" + str(n_pl)

    def save(self):
        utils.thin_pickle(self.save_path+"/metrics.pkl", self.n_pl, self.thin_freq)

        if self.verbose: print("thinned logs saved!")


# Keeping but not refactored to flax/optax yet
class maxlrCB(Callback):
    # sets LR
    def __init__(self, spectrumCB, lr_init=None, opt_cl=None, opt_alg=None, safe_const=None, warmup=0, est_ps=True,
                 early_stop=np.inf, save_freq=1, save_pref="traj/", verbose=False):
        super().__init__(save_freq=save_freq, save_pref=save_pref, verbose=verbose)
        self.safe_const = float(save_freq) if safe_const is None else safe_const
        self.warmup = warmup
        self.early_stop = early_stop
        self.spectrumCB = spectrumCB
        self.est_ps = est_ps
        self.opt_cl = opt_cl
        self.opt_alg = opt_alg
        self.name = "maxlr_" +str(self.safe_const)
        self.lrs = []
        self.last_new_lr = None
        if lr_init is not None:
            self.lrs.append(lr_init)
            self.last_new_lr = lr_init
        self.sharps = []
        self.best_sharp = -np.inf
        self.counter = 0
        self.alpha = -1.
        self.jit_optim = None

    def forward(self, **kwargs):
        epoch = kwargs['epoch']
        if epoch % self.save_freq == 0:

            iter_sharpness = self.spectrumCB.last_eigval[0] # get S0
            self.sharps.append(iter_sharpness)

            if self.best_sharp > 0 and self.est_ps:
                self.alpha = max(iter_sharpness - self.best_sharp, 1e-6) # get alpha

            if iter_sharpness > self.best_sharp:
                self.counter = 0
                self.best_sharp = iter_sharpness
            else:
                self.counter += 1

            if epoch >= self.warmup:

                if self.alpha < 0 or len(self.lrs) == 0:
                    lr_new = (2/iter_sharpness)/(self.safe_const) #
                else:
                    # lr_new = (2/(1.*min(iter_sharpness, self.best_sharp)+2.*self.alpha))
                    alpha_adj = self.alpha / self.last_new_lr
                    # s0 = max(iter_sharpness, self.best_sharp)
                    s0 = min(iter_sharpness*3, self.best_sharp*2) # conservative estimate as to avoid EoS from 3rd order terms
                    lr_new = (-s0 + np.sqrt(s0**2+16*alpha_adj))/(4*alpha_adj)
                    # lr_new = 2/(s0 + self.alpha)
                    # limit the amount that lr_new can grow
                    lr_new = min(self.lrs[-1]*2, lr_new)

                if len(self.lrs)==0 or np.abs(lr_new - self.last_new_lr) > 0.001:
                    optimisers = kwargs['optims']
                    model = kwargs['model']
                    if self.opt_alg is not None:
                        optimisers.append(tx.Optimizer(self.opt_alg(lr_new)))
                    else:
                        assert self.opt_cl is not None
                        optimisers.append(self.opt_cl((lr_new)))
                    optimisers[-1] = optimisers[-1].init(model.parameters())
                    if self.verbose: print(f"Hess {iter_sharpness:.2f}, LR changed to {lr_new:.3f}, epoch{epoch}, counter{self.counter}, ps{self.alpha}")
                    self.last_new_lr = lr_new
                else:
                    if self.verbose: print(f"Hess {iter_sharpness:.2f}, LR maintained at {self.last_new_lr:.3f}, epoch{epoch}, counter{self.counter}, ps{self.alpha}")

                self.lrs.append(lr_new)

            # update metrics
            # if self.best_sharp > 0 and self.est_ps:
            #     self.alpha = max(iter_sharpness - self.best_sharp, 0.)


    def check_es(self):
        return self.counter >= self.early_stop

        # print("Optimiser lr reduced")

    def save(self):
        utils.save_thing(np.array(self.lrs), self.save_path + f"/maxLR.pkl")
        print(f"Schedule of maxLR callback saved!")


# Keeping but not refactored to flax/optax yet
class reparamCB(Callback):
    #Dinh Reparam
    def __init__(self, epos, reparam, save_freq=0, save_pref="traj/", verbose=False):
        super().__init__(save_freq=save_freq, save_pref=save_pref, verbose=verbose)

        self.epos = epos
        self.reparam = reparam
        self.inverse = 1/np.array(self.reparam)
        self.vlines = None
        self.name = "reparam_" + str(hash(frozenset(epos)))
        for i in range(len(reparam)):
            self.name += "_" + str(hash(frozenset(reparam[i])))

    def forward(self, **kwargs):
        epoch = kwargs['epoch']
        if epoch in self.epos:
            ind = self.epos.index(epoch)
            reparam = self.reparam[ind]
            model = kwargs['model']
            flat_weights, _ = fu.ravel_pytree(model)
            vlines = utils.depth_vlines(model, layer=True)
            complete_vlines = np.zeros(len(vlines) + 1).astype(int)
            complete_vlines[1:] = vlines
            try:
                assert len(vlines) == len(reparam)
            except AssertionError:
                print(len(vlines), len(reparam))
                raise AssertionError
            new_weights = np.zeros_like(flat_weights)
            # print(flat_weights.shape)
            for i in range(len(reparam)):
                start, finish = complete_vlines[i], complete_vlines[i+1]
                new_weights[start:finish] = flat_weights[start:finish]*reparam[i]
            # new_weights = [flat_weights[complete_vlines[i]:complete_vlines[i+1]] * reparam[i] for i in range(len(flat_weights))]
            weight_updates = kwargs['weightsList']
            weight_updates.append(new_weights)

            # print("Optimiser lr reduced")

    def save(self):
        pass


class earlyStopCB(Callback):
    # sets LR
    def __init__(self, acc_threshold=None, max_eps=1999, min_eps=None, cbs=None, final_cbs=None, conseq_eps=1,
                 save_final_weights=True, save_freq=1, save_pref="traj/", verbose=False, low_thresh=0.11, low_eps=100):
        super().__init__(save_freq=save_freq, save_pref=save_pref, verbose=verbose)
        self.acc_threshold = acc_threshold
        self.cbs = cbs
        self.max_eps = max_eps
        self.min_eps = min_eps
        self.conseq_eps = conseq_eps
        self.sfw = save_final_weights
        self.final_cbs = final_cbs

        self.name = "esCB"
        if self.acc_threshold is not None:
            self.name += "_"+str(acc_threshold)
        if self.conseq_eps > 1:
            self.name += f"_conseq{conseq_eps}"
        if self.sfw:
            self.name += "_save"

        self.final_epoch = None
        self.final_model = None
        self.accs = np.zeros((low_eps))
        self.low_thresh = low_thresh
        self.low_eps = low_eps
        self.acc_counter = 0
        self.conseq_counter = 0

    def forward(self, **kwargs):
        if kwargs['epoch'] % self.save_freq != 0:
            return
        break_flag = True
        # don't allow break when under min eps requirement
        if self.min_eps is not None:
            if kwargs['epoch'] <= self.min_eps:
                break_flag = False

        # don't allow break when under train_acc requirement
        if self.acc_threshold is not None:
            if 'tr_acc' in kwargs:
                if self.verbose: print("ES", kwargs['tr_acc'], 'tr_acc' in kwargs)
                if math.isnan(kwargs['tr_acc']):
                    break_flag = True
                else:
                    if kwargs['tr_acc'] < self.acc_threshold:
                        break_flag = False
                        self.accs[self.acc_counter%self.low_eps] = kwargs['tr_acc']
                        self.acc_counter += 1
                    else:
                        if self.conseq_counter < self.conseq_eps:
                            self.conseq_counter += 1
                            break_flag = False

                # break after low eps (100) epochs of no improvement above baseline (low thresh)
                if self.acc_counter >= self.low_eps and np.mean(self.accs) < self.low_thresh:
                    break_flag = True
            else:
                break_flag = False

        # don't allow break when CB requirements are not fulfilled (currently only maxLRCB)
        if self.cbs is not None:
            for cb in self.cbs:
                if cb.check_es() != True:
                    break_flag=False

        if 'tr_loss' in kwargs:
            if math.isnan(kwargs['tr_loss']):
                break_flag = True

        if break_flag:
            # if self.acc_threshold is not None:
                # print("Train acc", kwargs['tr_acc'])
            self.final_epoch = kwargs['epoch']
            self.final_state = kwargs['state']
            for cb in self.final_cbs:
                cb.forward(epoch=0, state=kwargs['state'], train=kwargs['train'])

            return 'break'

    def save(self, error=False):
        if self.final_epoch is not None:
            utils.save_thing(np.zeros(1), self.save_path + f"/early_stop{self.final_epoch}.pkl")
            if self.final_state is not None and self.sfw:
                utils.save_weights(self.final_state, self.save_path + "/w" + str(self.final_epoch) + ".pkl")

            if error:
                utils.save_thing(np.zeros(1), self.save_path + f"/error{self.final_epoch}.pkl")
        else:
            utils.save_thing(np.zeros(1), self.save_path + f"/no_converge{self.max_eps}.pkl")
            if error:
                utils.save_thing(np.zeros(1), self.save_path + f"/error_nc_{self.max_eps}.pkl")


# Keeping but not refactored to flax/optax yet
class accLaCB(Callback):
    # sets LR
    def __init__(self, acc_threshs, lrs, opt_algs, opt_alg_names, save_freq=0, save_pref="traj/", verbose=False):
        super().__init__(save_freq=save_freq, save_pref=save_pref, verbose=verbose)
        self.acc_threshs = acc_threshs
        self.lrs = lrs
        self.opt_calls = opt_calls
        self.counter = 0
        assert len(acc_threshs) == len(lrs)
        assert len(acc_threshs) == len(opt_algs)
        self.name = "accLaSchedule_" + str(hash(frozenset(self.acc_thresh))) + "_" + str(hash(frozenset(self.lr_list)))+"_"+opt_alg_names

    def forward(self, **kwargs):
        # epoch = kwargs['epoch']
        acc = kwargs['tr_acc']
        if acc > self.acc_threshs[self.counter]:
            optimisers = kwargs['optims']
            model = kwargs['model']
            optimisers.append(self.opt_calls[self.counter](self.lrs[self.counter]))
            optimisers[-1] = optimisers[-1].init(model.parameters())
            if self.verbose: print("LR changed to", self.lrs[self.counter])
            self.counter += 1
            # print("Optimiser lr reduced")

    def save(self):
        pass

