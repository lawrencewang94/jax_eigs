# this came from VGG-SAM-variations.py
import numpy as np
from ml_collections import ConfigDict
import time

def make_fixed_configs():
    import jax.numpy as jnp
    import jax
    import optax
    # data configs
    data_config = ConfigDict(
        dict(
            n_train=4670,  # 2335, 4670, 9340, 2391884
            n_eval=276,  # 283287 total
            n_hess=5120,
            seq_len=1024,
            stride=1024,  # if half of seq len, then x2 data; 1024 is total sq len
            use_mse=False,
        )
    )

    # model configs
    model_config = ConfigDict(
        dict(
            arch_name='gpt2-mini',
            vocab_size=50257,
            hidden_size=384,  # 768 for small, 512 for mini
            num_layers=6,  # 12 for small, 4 for mini
            num_heads=6,  # 12 for small, 8 for mini
            head_dim=64,
            mlp_expansion=4,
            dropout_rate=0.1,
            max_seq_len=1024,  # 1024 for small, 512 for mini
            num_outputs=50257,
            dtype=jax.dtypes.bfloat16,  # jnp.float32, jax.dtypes.bfloat16
            causal_mask=True,
            softmax_dtype=jnp.float32,
            remat=[],  # "MLP", "Attn"
            scan_layers=False,
        )
    )
    # optimizer configs
    optim_config = ConfigDict(
        dict(
            lr=5e-4,
            bs=16,
            eval_bs=16,
            lr_decay_mode='cosine',
            force_fb=False,
            grad_accum=32,
            n_epochs=100_000,
            use_steps=True,
            warmup_steps=2000,
            loss_fn=optax.softmax_cross_entropy_with_integer_labels,
            b1=0.9,
            b2=0.95,
            b3=0.,
            sam=None,
            sam_rho=0.,
            sam_sync=1,
            wd=0.01,
            gn_clip=1.,
        )
    )
    # log configs
    log_config = ConfigDict(
        dict(
            tqdm_freq=1000,
            eval_freq=100,
            demo_outputs=False,
            dynamic_tqdm=False,
        )
    )
    # cb configs
    cb_config = ConfigDict(
        dict(
            sws=10_000,
            cb_freq=1,
            compute_hessian=False,
            hess_freq=1e8,
            n_eigs=20,
            n_evecs=10,
            use_es=False,
            es_stat='train_perplexity',
            es_mode='min',
            es_thresh=0.,
            es_consec=3,
            es_low=30.,
            es_min_eps=0,
            es_low_eps=60,
            use_bm=True,
            bm_stat='test_loss',
            bm_mode='min',
        )
    )
    config = ConfigDict(
        dict(
            model=model_config,
            optim=optim_config,
            data=data_config,
            log=log_config,
            cb=cb_config,
            force_train=False,
        )
    )

    return config


def train_model(var_cfg, resume=False, n_workers=8, gpu_id=None):
    # TODO write resume logic
    import typing as tp

    import jax
    import jax.numpy as jnp
    from flax import linen as nn  # Linen API
    import optax

    import lib_data
    import utils
    import modules
    import callbacks
    import sam_fam_optims as sfo
    import training

    from flax import struct  # Flax dataclasses
    from clu import metrics
    from perplexity import Perplexity
    from gpt2_utils import Transformer, load_params_gpt2mini, token_predictions
    from flax.training import train_state  # Useful dataclass to keep train state

    fixed_cfg = make_fixed_configs()
    # variable CFGs
    cfg = utils.deep_merge(fixed_cfg, var_cfg)
    if gpu_id is not None:
        cfg.gpu_id = gpu_id

    # cfg = deep_merge(fixed_cfg, var_cfg)

    # LOAD HYPS
    option = 'sam' if cfg.optim.sam is not None else ''

    # -----------------------------------------------------------------------------------------------------------------------------
    # FIXED PARAMS

    # -----------------------------------------------------------------------------------------------------------------------------
    # Datasets
    data_name = "OWT_all"

    def __get_datasets__():
        lib_data.preprocess_openwebtext()
        # train_dataset = lib_data.BinDataset("./tokenized_openwebtext/train.bin", block_size=cfg.data.seq_len)
        # val_dataset = lib_data.BinDataset("./tokenized_openwebtext/val.bin", block_size=cfg.data.seq_len)
        print("Datasets preprocessed")
        # return (train_dataset, val_dataset)
        return None

    # -----------------------------------------------------------------------------------------------------------------------------
    # Architecture

    def __get_arch__():

        model = Transformer(cfg)
        model_name = f"GPT2-mini-scratch"
        return model, model_name

    # -----------------------------------------------------------------------------------------------------------------------------
    # Optimizer
    # def __get_optim__(mode='constant'):
    #     if mode == 'constant':
    #         second_scheduler = optax.constant_schedule
    #     elif mode == 'cosine':
    #         second_scheduler = optax.cosine_decay_schedule
    #     elif mode == 'linear':
    #         second_scheduler = optax.linear_schedule
    #     else:
    #         raise NotImplementedError
    #
    #     base_string = f"1b{cfg.optim.b1}_2b{cfg.optim.b2}_3b{cfg.optim.b3}_lr{cfg.optim.lr}_warmup{cfg.optim.warmup_steps}"
    #     if cfg.optim.wd > 0:
    #         base_string += f"_wd{cfg.optim.wd}"
    #     if cfg.optim.gn_clip is not None:
    #         base_string += f"_GNclip{cfg.optim.gn_clip}"
    #     if (cfg.optim.sam is None) or cfg.optim.sam == '':
    #         warmup_scheduler = optax.linear_schedule(init_value=0.0, end_value=cfg.optim.lr,
    #                                                  transition_steps=cfg.optim.warmup_steps,
    #                                                  transition_begin=0, )
    #         constant_scheduler = optax.constant_schedule(cfg.optim.lr)
    #         lr_scheduler = optax.join_schedules([warmup_scheduler, constant_scheduler],
    #                                             boundaries=[cfg.optim.warmup_steps])
    #         optimizer = modules.get_sgd_optimizer(lr_scheduler, cfg.optim.b1, cfg.optim.b2, cfg.optim.b3, cfg.optim.wd,
    #                                               cfg.optim.gn_clip, verbose=False)
    #         optim_name = f"sgdFam_{base_string}"
    #
    #     elif cfg.optim.sam[-3:] == 'sam':
    #         assert cfg.optim.sam_rho is not None
    #         adv_lr = cfg.optim.sam_rho * cfg.optim.lr
    #         warmup_scheduler = optax.linear_schedule(init_value=0.0, end_value=cfg.optim.lr,
    #                                                  transition_steps=cfg.optim.warmup_steps,
    #                                                  transition_begin=0, )
    #         constant_scheduler = optax.constant_schedule(cfg.optim.lr)
    #         lr_scheduler = optax.join_schedules([warmup_scheduler, constant_scheduler],
    #                                             boundaries=[cfg.optim.warmup_steps])
    #
    #         adv_warmup_scheduler = optax.linear_schedule(init_value=0.0, end_value=adv_lr,
    #                                                      transition_steps=cfg.optim.warmup_steps,
    #                                                      transition_begin=0, )
    #         adv_constant_scheduler = optax.constant_schedule(adv_lr)
    #         adv_lr_scheduler = optax.join_schedules([adv_warmup_scheduler, adv_constant_scheduler],
    #                                                 boundaries=[cfg.optim.warmup_steps])
    #
    #         base_opt = modules.get_sgd_optimizer(lr_scheduler, cfg.optim.b1, cfg.optim.b2, cfg.optim.b3, cfg.optim.wd,
    #                                              cfg.optim.gn_clip, verbose=False)
    #         adv_opt = modules.get_sgd_optimizer(adv_lr_scheduler, cfg.optim.b1, cfg.optim.b2, cfg.optim.b3,
    #                                             cfg.optim.wd, cfg.optim.gn_clip, verbose=False)
    #
    #         if cfg.optim.sam == 'sam':
    #             optimizer = sfo.sam(base_opt, adv_opt, sync_period=cfg.optim.sam_sync, opaque_mode=True)  # sam opt
    #             optim_name = f"sgdFam-SAM_{base_string}_rho{cfg.optim.sam_rho}_syncT{cfg.optim.sam_sync}"
    #         elif cfg.optim.sam == 'asam':
    #             optimizer = sfo.asam(base_opt, adv_opt, sync_period=cfg.optim.sam_sync, opaque_mode=True)  # sam opt
    #             optim_name = f"sgdFam-aSAM_{base_string}_rho{cfg.optim.sam_rho}_syncT{cfg.optim.sam_sync}"
    #         if cfg.optim.sam == 'lsam':
    #             optimizer = sfo.looksam(base_opt, adv_opt, sync_period=cfg.optim.sam_sync, beta=0.9,
    #                                     opaque_mode=True)  # sam opt
    #             optim_name = f"sgdFam-lSAM_{base_string}_rho{cfg.optim.sam_rho}_syncT{cfg.optim.sam_sync}"
    #
    #     else:
    #         raise NotImplementedError
    #
    #     assert cfg.optim.grad_accum >= 1
    #     if cfg.optim.grad_accum > 1:
    #         optim_name += f"_ebs{cfg.optim.bs*cfg.optim.grad_accum}"
    #     else:
    #         optim_name += f"_bs{cfg.optim.bs}"
    #
    #     return optimizer, optim_name

    def __get_optim__(verbose=False):
        """
        Builds the optimizer and learning rate scheduler based on the config.
        """
        base_string = (
            f"1b{cfg.optim.b1}_2b{cfg.optim.b2}_3b{cfg.optim.b3}_lr{cfg.optim.lr}_"
            f"warmup{cfg.optim.warmup_steps}"
        )

        if cfg.optim.wd > 0:
            base_string += f"_wd{cfg.optim.wd}"
        if cfg.optim.gn_clip is not None:
            base_string += f"_GNclip{cfg.optim.gn_clip}"
        if cfg.optim.lr_decay_mode != 'constant':
            base_string += f"_{cfg.optim.lr_decay_mode}Decay"

        lr_scheduler = modules.build_lr_schedule(cfg, is_adv=False)

        if not cfg.optim.sam:
            optimizer = modules.get_sgd_optimizer(
                lr_scheduler,
                cfg.optim.b1, cfg.optim.b2, cfg.optim.b3,
                cfg.optim.wd, cfg.optim.gn_clip,
                verbose=verbose
            )
            optim_name = f"sgdFam_{base_string}"
        else:
            assert cfg.optim.sam_rho is not None
            adv_lr_scheduler = modules.build_lr_schedule(cfg, is_adv=True)

            base_opt = modules.get_sgd_optimizer(
                lr_scheduler,
                cfg.optim.b1, cfg.optim.b2, cfg.optim.b3,
                cfg.optim.wd, cfg.optim.gn_clip,
                verbose=verbose
            )
            adv_opt = modules.get_sgd_optimizer(
                adv_lr_scheduler,
                cfg.optim.b1, cfg.optim.b2, cfg.optim.b3,
                cfg.optim.wd, cfg.optim.gn_clip,
                verbose=verbose
            )

            if cfg.optim.sam == 'sam':
                optimizer = sfo.sam(base_opt, adv_opt, sync_period=cfg.optim.sam_sync, opaque_mode=True)
                optim_name = f"sgdFam-SAM_{base_string}_rho{cfg.optim.sam_rho}_syncT{cfg.optim.sam_sync}"
            elif cfg.optim.sam == 'asam':
                optimizer = sfo.asam(base_opt, adv_opt, sync_period=cfg.optim.sam_sync, opaque_mode=True)
                optim_name = f"sgdFam-aSAM_{base_string}_rho{cfg.optim.sam_rho}_syncT{cfg.optim.sam_sync}"
            elif cfg.optim.sam == 'lsam':
                optimizer = sfo.looksam(base_opt, adv_opt, sync_period=cfg.optim.sam_sync, beta=0.9, opaque_mode=True)
                optim_name = f"sgdFam-lSAM_{base_string}_rho{cfg.optim.sam_rho}_syncT{cfg.optim.sam_sync}"
            else:
                raise NotImplementedError(f"SAM variant {cfg.optim.sam} not supported.")

        # Add batch size info to name
        ebs = cfg.optim.bs * cfg.optim.grad_accum
        optim_name += f"_ebs{ebs}" if cfg.optim.grad_accum > 1 else f"_bs{cfg.optim.bs}"

        return optimizer, optim_name

    # -----------------------------------------------------------------------------------------------------------------------------
    # Callbacks
    def __get_cbs__(state,):
        cbs = []
        cbs.append(callbacks.saveWeightsCB(cfg.cb.sws, grad=True, skip0=True))

        if cfg.cb.compute_hessian:
            hvpCB = callbacks.hvpCB(loss_fn=cfg.optim.loss_fn,
                                    batches=(datasets[2].data[:cfg.data.n_hess], datasets[2].targets[:cfg.data.n_hess]),
                                    save_freq=cfg.cb.hess_freq, hess_bs=cfg.data.n_hess, state=state)
            cbs.append(hvpCB)
            specCB = callbacks.spectrumCB(n_eigs=cfg.cb.n_eigs, n_evecs=cfg.cb.n_evecs,
                                          loss_fn=cfg.optim.loss_fn, seed=cfg.optim.seed, hvpCB=hvpCB,
                                          save_freq=cfg.cb.hess_freq, verbose=False)
            cbs.append(specCB)

        if cfg.cb.use_es:
            esCB = callbacks.earlyStopCB(threshold=cfg.cb.es_thresh, cbs=None, min_eps=cfg.cb.es_min_eps,
                                         max_eps=cfg.optim.n_epochs, conseq_eps=cfg.cb.es_consec,
                                         final_cbs=[], verbose=False, low_eps=cfg.cb.es_low_eps,
                                         low_thresh=cfg.cb.es_low, mode=cfg.cb.es_mode, statistic=cfg.cb.es_stat)
            cbs.append(esCB)
        if cfg.cb.use_bm:
            bmCB = callbacks.bestModelCB(statistic=cfg.cb.bm_stat, mode=cfg.cb.bm_mode, save_freq=cfg.log.eval_freq)
            cbs.append(bmCB)
        return cbs

    # -----------------------------------------------------------------------------------------------------------------------------
    # Train State
    @struct.dataclass
    class Metrics(metrics.Collection):
        accuracy: metrics.Accuracy
        perplexity: Perplexity
        loss: metrics.Average.from_output('loss')

    class TrainState(train_state.TrainState):
        metrics: Metrics
        rng: jax.Array

    class TrainStateBN(train_state.TrainState):
        metrics: Metrics
        batch_stats: tp.Any
        rng: jax.Array

    class TrainStateSAM(modules.TrainStateSAM):
        metrics: Metrics
        batch_stats: tp.Any
        rng: jax.Array

    def create_train_state(model, optimizer, inputs, rng, option=""):
        """Creates an initial `TrainState`."""
        rng, model_rng = jax.random.split(rng)
        if option == "":
            params = model.init(model_rng, jnp.ones_like(inputs[0][jnp.newaxis, :]))[
                'params']  # initialize parameters by passing a template image

            tx = optimizer
            return TrainState.create(
                apply_fn=model.apply, params=params, tx=tx, metrics=Metrics.empty(), rng=rng)

        elif option == "bn":
            variables = model.init(model_rng, jnp.ones_like(
                inputs[0][jnp.newaxis, :]))  # initialize parameters by passing a template image
            params = variables['params']
            batch_stats = variables['batch_stats']

            tx = optimizer
            return TrainStateBN.create(
                apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats,
                metrics=Metrics.empty(), rng=rng)

        elif option == "sam":
            variables = model.init(model_rng, jnp.ones_like(
                inputs[0][jnp.newaxis, :]))  # initialize parameters by passing a template image
            params = variables['params']
            batch_stats = variables['batch_stats']

            tx = optimizer
            return TrainStateSAM.create(
                apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats,
                metrics=Metrics.empty(), rng=rng)
        else:
            raise NotImplementedError

    # -----------------------------------------------------------------------------------------------------------------------------
    # Training

    metrics_history = {'train_loss': [],
                   'train_accuracy': [],
                   'train_perplexity': [],
                   'test_loss': [],
                   'test_accuracy': [],
                   'test_perplexity': [],
                      }

    datasets = __get_datasets__()

    train_loader = lib_data.FastMemmapDataLoader(
        path="./tokenized_openwebtext/train.bin",
        block_size=cfg.data.seq_len,
        batch_size=cfg.optim.bs,
        buffer_size=100_000_000,
        shuffle=True,
        seed=cfg.optim.seed  # Fixes the order
    )

    # Load full val dataset into memory (e.g., token sequences)
    val_data = np.memmap("./tokenized_openwebtext/val.bin", dtype=np.uint16, mode="r")
    val_data = np.array(val_data)  # load fully into memory
    val_x, val_y = utils.chunk_into_sequences(val_data, cfg.data.seq_len)
    val_data = lib_data.Dataset(val_x, np.array(val_y).astype(jnp.int32))

    test_loader = lib_data.NumpyLoader(val_data, batch_size=cfg.optim.eval_bs, num_workers=n_workers)

    for sample_batch in test_loader:
        break
    print("sample test batch", sample_batch[0].shape, sample_batch[0].dtype, sample_batch[1].shape, sample_batch[1].dtype)

    dataloaders = [train_loader, test_loader]
    sample_batch = next(iter(train_loader))
    print("sample train batch", sample_batch[0].shape, sample_batch[1].shape)

    model, model_name = __get_arch__()
    model_name += "_seed" + str(cfg.optim.seed)

    optim, optim_name = __get_optim__(verbose=True)
    init_rng = jax.random.PRNGKey(cfg.optim.seed)
    state = create_train_state(model, optim, sample_batch[0], init_rng, option=option)
    del init_rng  # Must not be used anymore.

    # sample_out = state.apply_fn({'params': state.params,}, sample_batch[0][0][np.newaxis, :], train=False)
    sample_out = state.apply_fn({'params': state.params,}, sample_batch[0], train=False)
    print("sample out", sample_out.shape)

    print(utils.count_params(state.params))
    if cfg.log.demo_outputs: token_predictions(state, sample_batch)

    cbs = __get_cbs__(state)
    cb_name_str = utils.get_callback_name_str(cbs)
    cb_name_list = utils.get_callback_name_list(cbs)
    # break

    experiment_name = utils.get_now() + "_" + data_name + "_" + model_name + "_" + optim_name

    out_str = ""
    try:
        if cfg.force_train:
            raise FileNotFoundError

        experiment_name, lse = utils.find_latest_exp(experiment_name, cfg.optim.n_epochs,
                                                              cbs=cb_name_list, verbose=False)

        # experiment_name, lse = utils.find_latest_exp_no_epoch(experiment_name, max_eps=cfg.optim.n_epochs,
        #                                              cbs=cb_name_list, verbose=True)

        metrics_history = utils.load_thing("traj/" + experiment_name + "/metrics.pkl")
        print(f"tr_acc: {metrics_history['train_accuracy'][-1]:0%}, te_acc: {metrics_history['test_accuracy'][-1]:0%}")
        metrics_history['lse'] = [lse]
        if cfg.cb.compute_hessian:
            eigvals = utils.load_thing("traj/" + experiment_name + "/eigvals.pkl")
            metrics_history['eigvals'] = eigvals
            print(f"sharp: {metrics_history['eigvals'][-1][0]}")

    except FileNotFoundError:
        metrics_history, state = training.train_model(state, model, cfg.optim.loss_fn, metrics_history, cfg.optim.n_epochs, dataloaders, \
                                               experiment_name, cbs, option=option, force_fb=cfg.optim.force_fb, cfg=cfg,
                                               tqdm_over_epochs=cfg.log.tqdm_freq, eval_freq=cfg.log.eval_freq, gradient_accumulation=cfg.optim.grad_accum,
                                                      return_state=True, steps_not_epochs=cfg.optim.use_steps, dynamic_bar=cfg.log.dynamic_tqdm)

    if cfg.log.demo_outputs: token_predictions(state, sample_batch)

    out_str += f"tr_acc: {metrics_history['train_accuracy'][-1]:0%}, te_acc: {metrics_history['test_accuracy'][-1]:0%}"
    if cfg.cb.compute_hessian:
        eigvals = utils.load_thing("traj/" + experiment_name + "/eigvals.pkl")
        metrics_history['eigvals'] = eigvals
        out_str += f"sharp: {metrics_history['eigvals'][-1][0]}"

    metrics_history['experiment_name'] = experiment_name

    return out_str, metrics_history,
