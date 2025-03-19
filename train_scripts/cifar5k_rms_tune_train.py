# this came from VGG-SAM-variations.py
def train_model(hyp, supplementary=None, resume=False, load_only=False, n_workers=8):
    # TODO write resume logic
    import typing as tp

    import jax
    import jax.numpy as jnp
    from flax import linen as nn  # Linen API
    import optax
    import torch
    import numpy as np

    import lib_data
    import utils
    import modules
    import callbacks
    import sam_fam_optims as sfo
    import training

    from flax import struct  # Flax dataclasses
    from clu import metrics
    from flax.training import train_state  # Useful dataclass to keep train state

    # LOAD HYPS
    seed = hyp[0]  # was 0
    arch_name = hyp[1]  # was 1
    bs = hyp[2]  # was 2
    lr, b1, b2, b3 = hyp[3]  # was 3
    option = 'bn'

    # -----------------------------------------------------------------------------------------------------------------------------
    # FIXED PARAMS

    # data params
    n_out = 10
    n_train: int = 512 * n_out
    n_eval: int = 200 * n_out
    n_hess: int = 512 * n_out
    use_mse = False

    # optim params
    warmup_steps = 2
    eval_bs = 2000
    max_epochs = 5000
    sam_type, sam_rho, sam_sync = None, 0., 1.
    loss_fn = optax.softmax_cross_entropy_with_integer_labels

    # callback params
    save_weight_freq = 10
    cb_freq = 1
    hess_freq = int(1e8)  # really large

    # training params
    compute_hessian = True
    force_train = False
    force_fb = False
    tqdm_freq = 100

    # -----------------------------------------------------------------------------------------------------------------------------
    # Datasets
    data_name = "cifar10_" + str(n_out) + "cl_" + str(n_train) + "_" + str(n_eval)

    def __get_datasets__():
        datasets = lib_data.get_cifar10(flatten=False, tr_indices=n_train, te_indices=n_eval, hess_indices=n_hess,
                                        tr_classes=n_out, te_classes=n_out, hess_classes=n_out, one_hot=use_mse,
                                        augmentations=False, visualise=True)

        return datasets

    # -----------------------------------------------------------------------------------------------------------------------------
    # Architecture
    def __get_arch__(arch_name, name_only=False):
        if arch_name == 'vgg':
            n_blocks = 3
            layers_per_block = 3
            base_width = 8
            use_DO = False
            use_BN = True
            depth_name = str(int(1 + n_blocks * layers_per_block))

            model_name = f"VGG{depth_name}_base{base_width:d}"
            if use_DO:
                model_name += "_DO"
            if use_BN:
                model_name += "_BN"

            if name_only:
                return None, model_name

            model = modules.VGGNet(n_blocks, layers_per_block, base_width, use_DO=use_DO, use_BN=use_BN)

            return model, model_name

        elif arch_name == 'resnet':
            n_blocks = 3
            layers_per_block = 3
            depth_per_layer = 2
            base_width = 8
            use_DO = False
            use_BN = True
            sc_conv = "Identity"
            depth_name = str(int(2 + n_blocks * layers_per_block * depth_per_layer))

            model_name = f"ResNet{depth_name}_base{base_width:d}_{sc_conv}"
            if use_DO:
                model_name += "_DO"
            if use_BN:
                model_name += "_BN"

            if name_only:
                return None, model_name

            model = modules.ResNet(n_blocks, layers_per_block, depth_per_layer, use_DO=use_DO, use_BN=use_BN,
                                   sc_conv=sc_conv)

            return model, model_name

        else:
            return NotImplementedError

    # -----------------------------------------------------------------------------------------------------------------------------
    # Optimizer
    def __get_optim__(warmup_steps, lr, b1, b2, b3, sam_type="", rho=None, sync_period=1, name_only=False):
        # warmup_steps, lr, b1, b2, b3 = hyps['warmup_steps'], hyps['lr'], hyps['b1'], hyps['b2'], hyps['b3']
        if (sam_type is None) or sam_type == '':
            optim_name = f"sgdFam_1b{b1}_2b{b2}_3b{b3}_lr{lr}_warmup{warmup_steps}"
            if name_only:
                return None, optim_name
            warmup_scheduler = optax.linear_schedule(init_value=0.0, end_value=lr,
                                                     transition_steps=warmup_steps,
                                                     transition_begin=0, )
            constant_scheduler = optax.constant_schedule(lr)
            lr_scheduler = optax.join_schedules([warmup_scheduler, constant_scheduler], boundaries=[warmup_steps])
            optimizer = modules.get_sgd_optimizer(lr_scheduler, b1, b2, b3, verbose=False)

        elif sam_type[-3:] == 'sam':
            assert rho is not None

            if sam_type == 'sam':
                optim_name = f"sgdFam-SAM_1b{b1}_2b{b2}_3b{b3}_lr{lr}_warmup{warmup_steps}_rho{rho}_syncT{sync_period}"
            elif sam_type == 'asam':
                optim_name = f"sgdFam-aSAM_1b{b1}_2b{b2}_3b{b3}_lr{lr}_warmup{warmup_steps}_rho{rho}_syncT{sync_period}"
            if sam_type == 'lsam':
                optim_name = f"sgdFam-lSAM_1b{b1}_2b{b2}_3b{b3}_lr{lr}_warmup{warmup_steps}_rho{rho}_syncT{sync_period}"

            if name_only:
                return None, optim_name
            adv_lr = rho * lr
            warmup_scheduler = optax.linear_schedule(init_value=0.0, end_value=lr,
                                                     transition_steps=warmup_steps,
                                                     transition_begin=0, )
            constant_scheduler = optax.constant_schedule(lr)
            lr_scheduler = optax.join_schedules([warmup_scheduler, constant_scheduler], boundaries=[warmup_steps])

            adv_warmup_scheduler = optax.linear_schedule(init_value=0.0, end_value=adv_lr,
                                                         transition_steps=warmup_steps,
                                                         transition_begin=0, )
            adv_constant_scheduler = optax.constant_schedule(adv_lr)
            adv_lr_scheduler = optax.join_schedules([adv_warmup_scheduler, adv_constant_scheduler],
                                                    boundaries=[warmup_steps])

            base_opt = modules.get_sgd_optimizer(lr_scheduler, b1, b2, b3, verbose=False)
            adv_opt = modules.get_sgd_optimizer(adv_lr_scheduler, b1, b2, b3, verbose=False)

            if sam_type == 'sam':
                optimizer = sfo.sam(base_opt, adv_opt, sync_period=sync_period, opaque_mode=True)  # sam opt
                optim_name = f"sgdFam-SAM_1b{b1}_2b{b2}_3b{b3}_lr{lr}_warmup{warmup_steps}_rho{rho}_syncT{sync_period}"
            elif sam_type == 'asam':
                optimizer = sfo.asam(base_opt, adv_opt, sync_period=sync_period, opaque_mode=True)  # sam opt
                optim_name = f"sgdFam-aSAM_1b{b1}_2b{b2}_3b{b3}_lr{lr}_warmup{warmup_steps}_rho{rho}_syncT{sync_period}"
            if sam_type == 'lsam':
                optimizer = sfo.looksam(base_opt, adv_opt, sync_period=sync_period, beta=0.9,
                                        opaque_mode=True)  # sam opt
                optim_name = f"sgdFam-lSAM_1b{b1}_2b{b2}_3b{b3}_lr{lr}_warmup{warmup_steps}_rho{rho}_syncT{sync_period}"

        else:
            raise NotImplementedError

        return optimizer, optim_name

    # -----------------------------------------------------------------------------------------------------------------------------
    # Callbacks
    def __get_cbs__(state, compute_hessian=False):
        cbs = []
        cbs.append(callbacks.saveWeightsCB(save_weight_freq, grad=True))

        if compute_hessian:
            hvpCB = callbacks.hvpCB(loss_fn=loss_fn, batches=(datasets[2].data[:n_hess], datasets[2].targets[:n_hess]),
                                    save_freq=hess_freq, hess_bs=n_hess, state=state)
            cbs.append(hvpCB)
            specCB = callbacks.spectrumCB(n_eigs=20, n_evecs=10,
                                          loss_fn=loss_fn, seed=seed, hvpCB=hvpCB, save_freq=hess_freq, verbose=False)
            cbs.append(specCB)

        esCB = callbacks.earlyStopCB(acc_threshold=0.999, cbs=None, min_eps=save_weight_freq, max_eps=max_epochs,
                                     conseq_eps=2,
                                     final_cbs=[hvpCB, specCB], verbose=False, low_eps=max(save_weight_freq, 100),
                                     low_thresh=0.11, )
        cbs.append(esCB)
        return cbs

    # -----------------------------------------------------------------------------------------------------------------------------
    # Train State
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
            params = model.init(rng, jnp.ones_like(inputs[0][jnp.newaxis, :]))[
                'params']  # initialize parameters by passing a template image

            tx = optimizer
            return TrainState.create(
                apply_fn=model.apply, params=params, tx=tx, metrics=Metrics.empty())

        elif option == "bn":
            variables = model.init(rng, jnp.ones_like(
                inputs[0][jnp.newaxis, :]))  # initialize parameters by passing a template image
            params = variables['params']
            batch_stats = variables['batch_stats']

            tx = optimizer
            return TrainState.create(
                apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats,
                metrics=Metrics.empty())

        elif option == "sam":
            variables = model.init(rng, jnp.ones_like(
                inputs[0][jnp.newaxis, :]))  # initialize parameters by passing a template image
            params = variables['params']
            batch_stats = variables['batch_stats']

            tx = optimizer
            return TrainStateSAM.create(
                apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats,
                metrics=Metrics.empty())
        else:
            raise NotImplementedError

    # -----------------------------------------------------------------------------------------------------------------------------
    # Training

    metrics_history = {'train_loss': [],
                       'train_accuracy': [],
                       'test_loss': [],
                       'test_accuracy': []}

    if 'datasets' not in supplementary:
        datasets = __get_datasets__()
        supplementary['datasets'] = datasets
    else:
        datasets = supplementary['datasets']

    if load_only and 'cb_name_list' in supplementary:

        model, model_name = __get_arch__(arch_name, name_only=True)
        model_name += "_seed" + str(seed)

        optim, optim_name = __get_optim__(warmup_steps, lr, b1, b2, b3, sam_type=sam_type, rho=sam_rho,
                                          sync_period=sam_sync, name_only=True)
        optim_name += f"_bs{bs}"
        cb_name_list = supplementary['cb_name_list']

    else:

        model, model_name = __get_arch__(arch_name, name_only=False)
        model_name += "_seed" + str(seed)

        optim, optim_name = __get_optim__(warmup_steps, lr, b1, b2, b3, sam_type=sam_type, rho=sam_rho,
                                          sync_period=sam_sync, name_only=False)
        optim_name += f"_bs{bs}"

        torch.manual_seed(seed)
        train_loader = lib_data.NumpyLoader(datasets[0], batch_size=bs, shuffle=True, num_workers=n_workers)
        for sample_batch in train_loader:
            break

        if f'first_sample{seed}' in supplementary:
            assert supplementary[f'first_sample{seed}'] == np.array(sample_batch[0]).ravel()[0]
        else:
            supplementary[f'first_sample{seed}'] = np.array(sample_batch[0]).ravel()[0]
            # print(f'first_sample_seed:{seed}, value:{np.array(sample_batch[0]).ravel()[0]}')

        test_loader = lib_data.NumpyLoader(datasets[1], batch_size=eval_bs, num_workers=n_workers)
        dataloaders = [train_loader, test_loader]

        init_rng = jax.random.PRNGKey(seed)
        state = create_train_state(model, optim, sample_batch[0], init_rng, option=option)
        del init_rng  # Must not be used anymore.

        cbs = __get_cbs__(state, compute_hessian=compute_hessian)
        cb_name_str = utils.get_callback_name_str(cbs)
        cb_name_list = utils.get_callback_name_list(cbs)
        supplementary['cb_name_list'] = cb_name_list
    # break

    experiment_name = utils.get_now() + "_" + data_name + "_" + model_name + "_" + optim_name

    out_str = ""
    try:
        if force_train:
            raise FileNotFoundError
        experiment_name, lse = utils.find_latest_exp_no_epoch(experiment_name, max_eps=max_epochs,
                                                     cbs=cb_name_list, verbose=False)
        metrics_history = utils.load_thing("traj/" + experiment_name + "/metrics.pkl")
        metrics_history['lse'] = [lse]

    except FileNotFoundError:
        if load_only:
            return f"{experiment_name} not found. ", {}, supplementary

        metrics_history = training.train_model(state, model, loss_fn, metrics_history, max_epochs, dataloaders, \
                                               experiment_name, cbs, option=option, force_fb=force_fb,
                                               tqdm_over_epochs=False)

    out_str += f"tr_acc: {metrics_history['train_accuracy'][-1]:0.2%}, te_acc: {metrics_history['test_accuracy'][-1]:0.2%}"

    if compute_hessian:
        eigvals = utils.load_thing("traj/" + experiment_name + "/eigvals.pkl")
        metrics_history['eigvals'] = eigvals
        out_str += f", sharp: {metrics_history['eigvals'][-1][0]:.3E}"

    return out_str, metrics_history, supplementary
