#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.chdir("../")
print("Moved up")


# In[2]:


import importlib
import copy
import pickle
import typing as tp
from functools import partial

import jax
import jax.numpy as jnp
import jax.flatten_util as fu
from flax import linen as nn  # Linen API
import numpy as np
import matplotlib.pyplot as plt
import optax
import math

from tqdm import tqdm
import time

import lib_data
import utils
import modules
import callbacks

get_ipython().run_line_magic('env', 'XLA_PYTHON_CLIENT_MEM_FRACTION=.9')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

print("devices", jax.devices())

shade_colours = plt.get_cmap('Set3')
dark_colours = plt.get_cmap('tab10')
all_colours = plt.get_cmap('tab20')

def light_colours(i):
    return all_colours(2*i+1)


# In[ ]:


# %env XLA_FLAGS=--xla_gpu_deterministic_ops=true
# %env XLA_FLAGS=--xla_gpu_deterministic_reductions=true
get_ipython().run_line_magic('env', 'XLA_FLAGS=--xla_gpu_deterministic_ops=true --xla_gpu_deterministic_reductions=true')


# # Dataset

# In[3]:


#-----------------------------------------------------------------------------------------------------------------------------
n_out = 10

n_train: int = 512 * n_out
n_eval: int = 200 * n_out
n_hess: int = 512 * n_out
use_mse = False

importlib.reload(lib_data)
def __get_datasets():
    datasets = lib_data.get_cifar10(flatten=False, tr_indices=n_train, te_indices=n_eval, hess_indices=n_hess,
                            tr_classes=n_out, te_classes=n_out, hess_classes=n_out, one_hot = use_mse, 
                              augmentations = False, visualise=True)

    data_name = "cifar10_"+str(n_out)+"cl_"+str(n_train) + "_" + str(n_eval) 
    print(len(datasets))
    return datasets, data_name

datasets, data_name = __get_datasets()


# # Architecture

# In[4]:


#-----------------------------------------------------------------------------------------------------------------------------
importlib.reload(modules)

class VGGBlock(nn.Module):
    out_channels: int
    strides: int = 1
    depth: int = 2
    p_drop: float = 0.1
    dropout: bool = True
    kernel: tuple = (3,3)
    bn: bool = True
    pool: str = "max"
    default_kernel_init: tp.Callable = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, x, train: bool = True):
        for i in range(self.depth):
            x = nn.Conv(self.out_channels, self.kernel, strides=self.strides, kernel_init=self.default_kernel_init)(x)
            if self.bn:
                x = nn.BatchNorm(use_running_average=not train)(x)
            if self.dropout:
                x = nn.Dropout(self.p_drop)(x, deterministic=not train)
            x = nn.relu(x)  # Activation
        if self.pool == "max":
            x = modules.max_pool(x, (2, 2), strides=(2, 2))
        elif self.pool == "ave":
            x = modules.ave_pool(x, (2, 2), strides=(2, 2))
        else:
            raise NotImplementedError
            
        return x


class VGGNet(nn.Module):
    """A VGG model."""
    n_blocks: int = 3
    layers_per_block: int = 3
    base_width: int = 8
    width_expansion_factor: int = 2
    n_out: int = 10
    pool: str = 'max'

    use_DO: bool = False
    use_BN: bool = True
    deterministic: tp.Optional[bool] = None

    @nn.compact
    def __call__(self, x, train=True):

        for i in range(self.n_blocks):
            width_factor = self.width_expansion_factor ** i
            x = VGGBlock(out_channels=width_factor*self.base_width, depth=self.layers_per_block,
                        pool=self.pool, dropout=self.use_DO, bn=self.use_BN)(x, train)

        x = x.reshape(x.shape[0], -1) # flatten
        x = nn.Dense(width_factor)(x)
        if self.use_BN:
            x = nn.BatchNorm(use_running_average=not train)(x)
        if self.use_DO:
            x = nn.Dropout(self.p_drop)(x, deterministic=not train)
        x = nn.relu(x)
        x = nn.Dense(self.n_out)(x)

        return x


# In[5]:


def __get_arch__():
    n_blocks = 3
    layers_per_block = 3
    base_width = 8
    use_DO = False
    use_BN = True

    depth_name = str(int(1+n_blocks*layers_per_block))
    model = VGGNet(n_blocks, layers_per_block, base_width, use_DO=use_DO, use_BN=use_BN)
    model_name = f"VGG{depth_name}_base{base_width:d}"
    if use_DO:
        model_name += "_DO"
    if use_BN:
        model_name += "_BN"
    return model, model_name
    
model_arch, model_name = __get_arch__()
print(model_name)


# # Optimizer

# In[30]:


#-----------------------------------------------------------------------------------------------------------------------------
import sam_fam_optims as sfo
importlib.reload(modules)
importlib.reload(sfo)
from optax import contrib


def __get_optim__(warmup_steps, lr, b1, b2, b3, sam_type="", rho=None, sync_period=1):
    # warmup_steps, lr, b1, b2, b3 = hyps['warmup_steps'], hyps['lr'], hyps['b1'], hyps['b2'], hyps['b3']
    if sam_type[-3:] == 'sam':
        assert rho is not None
        adv_lr = rho*lr
        warmup_scheduler = optax.linear_schedule(init_value=0.0, end_value=lr,
                                                transition_steps=warmup_steps,
                                                transition_begin=0,)
        constant_scheduler = optax.constant_schedule(lr)
        lr_scheduler = optax.join_schedules([warmup_scheduler, constant_scheduler], boundaries=[warmup_steps])

        adv_warmup_scheduler = optax.linear_schedule(init_value=0.0, end_value=adv_lr,
                                                transition_steps=warmup_steps,
                                                transition_begin=0,)
        adv_constant_scheduler = optax.constant_schedule(adv_lr)
        adv_lr_scheduler = optax.join_schedules([adv_warmup_scheduler, adv_constant_scheduler], boundaries=[warmup_steps])
        
        base_opt = modules.get_sgd_optimizer(lr_scheduler, b1, b2, b3, verbose=False)
        adv_opt = modules.get_sgd_optimizer(adv_lr_scheduler, b1, b2, b3, verbose=False)
        
        if sam_type == 'sam':
            optimizer = sfo.sam(base_opt, adv_opt, sync_period=sync_period, opaque_mode=True) # sam opt
            optim_name = f"sgdFam-SAM_1b{b1}_2b{b2}_3b{b3}_lr{lr}_warmup{warmup_steps}_rho{rho}_syncT{sync_period}"
        elif sam_type == 'asam':
            optimizer = sfo.asam(base_opt, adv_opt, sync_period=sync_period, opaque_mode=True) # sam opt
            optim_name = f"sgdFam-aSAM_1b{b1}_2b{b2}_3b{b3}_lr{lr}_warmup{warmup_steps}_rho{rho}_syncT{sync_period}"
        if sam_type == 'looksam':
            optimizer = sfo.looksam(base_opt, adv_opt, sync_period=sync_period, beta=0.9, opaque_mode=True) # sam opt
            optim_name = f"sgdFam-lSAM_1b{b1}_2b{b2}_3b{b3}_lr{lr}_warmup{warmup_steps}_rho{rho}_syncT{sync_period}"
        
    else:
        warmup_scheduler = optax.linear_schedule(init_value=0.0, end_value=lr,
                                                transition_steps=warmup_steps,
                                                transition_begin=0,)
        constant_scheduler = optax.constant_schedule(lr)
        lr_scheduler = optax.join_schedules([warmup_scheduler, constant_scheduler], boundaries=[warmup_steps])
        optimizer = modules.get_sgd_optimizer(lr_scheduler, b1, b2, b3, verbose=False)
        optim_name = f"sgdFam_1b{b1}_2b{b2}_3b{b3}_lr{lr}_warmup{warmup_steps}"
        
    return optimizer, optim_name

optimizer, optim_name = __get_optim__(2, 0.1, 0, 0, 0)


# # Model Params (Fixed and Tuned)

# In[27]:


#-----------------------------------------------------------------------------------------------------------------------------
import itertools
### FIXED
warmup_steps = 2
bs = 64
eval_bs = 2000
n_epochs = 20
loss_fn = optax.softmax_cross_entropy_with_integer_labels

# FLEXIBLE

# lr, b1, b2, b3,
sgd_hp_list = [
    # (0.1, 0., 0., 0.), # sgd
    # (5e-3, 0., 0.99, 0.), # rms
    # (5e-3, 0., 0.99, -1.), # rms-UB
    (5e-3, 0.9, 0.99, 0.), # adam
    # (5e-3, 0.9, 0.99, -1.), # adam-UB
]

# type, rho (as a scaling factor of LR), sync_period
sam_hp_list = [
    # (None, 0., 1), # None
    # ('sam', 0., 1), # SAM reproduce
    ('sam', 1., 2), # SAM normal
    # ('sam', 1., 2), # SAM sync=2
    # ('asam', 0., 1), # SAM reproduce
    ('asam', 1., 2), # SAM normal
    # ('asam', 1., 2), # SAM sync=2
    # ('looksam', 0., 1), # SAM reproduce
    ('looksam', 1., 2), # SAM normal
    # ('looksam', 1., 2), # SAM sync=2
]

seed_list = [x for x in range(1)]
# seed_list = [1352] # just to check new SAM with new loss_wrap
# sam_list = [False, True]
s = [sgd_hp_list, sam_hp_list, seed_list]
hyp_list = list(itertools.product(*s))
print(len(hyp_list))


# # Callbacks

# In[13]:


#-----------------------------------------------------------------------------------------------------------------------------
sws = 5
cb_freq = 1
hess_freq = int(1e8) # really large
importlib.reload(callbacks)
def __get_cbs__(state, compute_hessian=False):
    cbs = []
    cbs.append(callbacks.saveWeightsCB(sws, grad=True))
    # cbs.append(callbacks.thinCB(thin_freq=cb_freq))

    if compute_hessian:
        hvpCB = callbacks.hvpCB(loss_fn=loss_fn, batches=(datasets[2].data[:n_hess], datasets[2].targets[:n_hess]), 
                            save_freq=hess_freq, hess_bs=n_hess, state=state)
        cbs.append(hvpCB)   
        specCB = callbacks.spectrumCB(n_eigs=20, n_evecs=10, 
                    loss_fn=loss_fn, seed=seed, hvpCB=hvpCB, save_freq=hess_freq, verbose=False)
        cbs.append(specCB)

    esCB = callbacks.earlyStopCB(acc_threshold=0.999, cbs=None, min_eps=sws, max_eps=n_epochs, conseq_eps=2,
                                 final_cbs=[hvpCB, specCB], verbose=False, low_eps=max(sws, 100), low_thresh=0.11, )
    cbs.append(esCB)
    return cbs


# # Train State

# In[14]:


from flax import struct                # Flax dataclasses
from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
importlib.reload(modules)

@struct.dataclass
class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')

class TrainState(train_state.TrainState):
    metrics: Metrics
    batch_stats: tp.Any

class TrainStateSAM(modules.TrainStateSAM):
    metrics: Metrics
    batch_stats: tp.Any

def create_train_state(model, optimizer, inputs, rng, option=""):
    """Creates an initial `TrainState`."""
    if option == "":
        params = model.init(rng, jnp.ones_like(inputs[0][jnp.newaxis, :]))['params'] # initialize parameters by passing a template image
        
        tx = optimizer
        return TrainState.create(
          apply_fn=model.apply, params=params, tx=tx, metrics=Metrics.empty())
        
    elif option == "bn":
        variables = model.init(rng, jnp.ones_like(inputs[0][jnp.newaxis, :])) # initialize parameters by passing a template image
        params = variables['params']
        batch_stats = variables['batch_stats']
        
        tx = optimizer
        return TrainState.create(
          apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats, 
          metrics=Metrics.empty())
        
    elif option == "sam":
        variables = model.init(rng, jnp.ones_like(inputs[0][jnp.newaxis, :])) # initialize parameters by passing a template image
        params = variables['params']
        batch_stats = variables['batch_stats']
        
        tx = optimizer
        return TrainStateSAM.create(
          apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats, 
          metrics=Metrics.empty())
    else:
        raise NotImplementedError
        
        


# # Training

# In[32]:


#-----------------------------------------------------------------------------------------------------------------------------
import training
importlib.reload(training)

load_files = False
compute_hessian = True
force_train = False

all_mh = []
all_exp_names = []

for hyp in hyp_list:
    
    metrics_history = {'train_loss': [],
                   'train_accuracy': [],
                   'test_loss': [],
                   'test_accuracy': []}

    lr, b1, b2, b3 = hyp[0]
    sam_type, sam_rho, sam_sync = hyp[1]
    # lr, b1, b2, b3, sam, sam_rho, sync_T = hyp[0]
    seed = hyp[2]
    option = 'sam' if sam_type is not None else 'bn'
    
    if datasets is None:
        datasets, data_name = __get_datasets__()
    
    train_loader = lib_data.NumpyLoader(datasets[0], batch_size=bs, shuffle=True)
    for sample_batch in train_loader:
        break
    
    test_loader = lib_data.NumpyLoader(datasets[1], batch_size=eval_bs)
    dataloaders = [train_loader, test_loader]
    
    model, model_name = __get_arch__()
    model_name += "_seed"+str(seed)

    optim, optim_name = __get_optim__(warmup_steps, lr, b1, b2, b3, sam_type=sam_type, rho=sam_rho, sync_period=sam_sync)
    optim_name += f"_epochs{n_epochs}_bs{bs}"

    init_rng = jax.random.PRNGKey(seed)
    state = create_train_state(model, optim, sample_batch[0], init_rng, option=option)
    del init_rng  # Must not be used anymore.
    
    cbs = __get_cbs__(state, compute_hessian=compute_hessian)
    cb_name_str = utils.get_callback_name_str(cbs)
    cb_name_list = utils.get_callback_name_list(cbs)
    # break
    num_params = utils.count_params(state.params)
    print("num params", num_params)

    experiment_name = utils.get_now() + "_" + data_name + "_" + model_name + "_" + optim_name
    
    try:
        if force_train:
            raise FileNotFoundError
        experiment_name, lse = utils.find_latest_exp(experiment_name, n_epochs, save_freq=cb_freq, 
                                                   cbs=cb_name_list, unknown_lse=True, verbose=False)
        metrics_history = utils.load_thing("traj/" + experiment_name + "/metrics.pkl")
        print(f"tr_acc: {metrics_history['train_accuracy'][-1]:0%}, te_acc: {metrics_history['test_accuracy'][-1]:0%}")
        metrics_history['lse'] = [lse]
        if compute_hessian:
            eigvals = utils.load_thing("traj/" + experiment_name + "/eigvals.pkl")
            metrics_history['eigvals'] = eigvals
            print(f"sharp: {metrics_history['eigvals'][-1][0]}")

    except FileNotFoundError:
        metrics_history = training.train_model(state, model, loss_fn, metrics_history, n_epochs, dataloaders, \
                                                   experiment_name, cbs, option=option, force_fb=False, tqdm_over_epochs=5)         
        
    all_mh.append(metrics_history)
    all_exp_names.append(experiment_name)
    
    print(experiment_name, "complete")
    print("\n ---------------------------------------------------------------------------------------------------------\n")
# Training: datasets, hps, arch_func, optim_func, cb_func, -> train model


# In[20]:


print(len(optim_hp_list), len(all_mh))


# # Results

# In[29]:


optim_names = ['SGD', 'SGD-SAM', 'ADAM', 'ADAM-SAM-R0', 'ADAM-SAM', 'ADAM-UB-1e0', 'ADAM-UB-1e0-SAM', 'ADAM-UB-5e-1','ADAM-UB-1e-1', 'ADAM-UB-5e-2', 'ADAM-UB-1e-2', 'ADAM-UB-5e-3','ADAM-UB-1e-3']
stat_names = ['train_accuracy', 'test_accuracy', 'lse']
n_seeds = len(seed_list)
for i in range(len(optim_hp_list)):
    stats = [ 0 for j in range(len(stat_names))]
    for j in range(len(stats)):
        for k in range(n_seeds*i, n_seeds*i+n_seeds):
            stats[j] += all_mh[k][stat_names[j]][-1]
    out = f"{optim_names[i]}"
    for j in range(len(stats)):
        out += f", {stat_names[j]}:{stats[j]/n_seeds}"
    print(out)
    # print(optim_hp_list[i], np.mean(all_mh[3*i:3*i+3]['train_accuracy'][-1]), np.mean(all_mh[3*i:3*i+3]['test_accuracy'][-1]))


# # Analysis

# In[ ]:





# In[ ]:





# In[ ]:




