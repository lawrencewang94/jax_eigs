# this came from VGG-SAM-variations.py
from ml_collections import ConfigDict

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
            seq_len=512,
            stride=512,  # if half of seq len, then x2 data; 1024 is total sq len
            use_mse=False,
        )
    )

    # model configs
    model_config = ConfigDict(
        dict(
            arch_name='gpt2',
            vocab_size=50257,
            hidden_size=512,  # 768 for small, 512 for mini
            num_layers=4,  # 12 for small, 4 for mini
            num_heads=8,  # 12 for small, 8 for mini
            head_dim=64,
            mlp_expansion=4,
            dropout_rate=0.1,
            max_seq_len=512,  # 1024 for small, 512 for mini
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
            lr=5e-5,
            bs=64,
            eval_bs=64,
            force_fb=False,
            grad_accum=1,
            n_epochs=30,
            warmup_steps=2,
            loss_fn=optax.softmax_cross_entropy_with_integer_labels,
            b1=0.9,
            b2=0.99,
            b3=0.,
            sam=None,
            sam_rho=0.,
            sam_sync=1,
            wd=0.01,
            gn_clip=None,
        )
    )
    # log configs
    log_config = ConfigDict(
        dict(
            tqdm_freq=1,
            eval_freq=1,
            demo_outputs=False,
        )
    )
    # cb configs
    cb_config = ConfigDict(
        dict(
            sws=1e8,
            cb_freq=1,
            compute_hessian=False,
            hess_freq=1e8,
            n_eigs=20,
            n_evecs=10,
            use_es=True,
            es_stat='train_perplexity',
            es_mode='min',
            es_thresh=0.,
            es_consec=3,
            es_low=30.,
            es_min_eps=0,
            es_low_eps=60,
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


def train_model(var_cfg, datasets=None, resume=False, n_workers=8):
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

    # cfg = deep_merge(fixed_cfg, var_cfg)

    # LOAD HYPS
    option = 'sam' if cfg.optim.sam is not None else ''

    # -----------------------------------------------------------------------------------------------------------------------------
    # FIXED PARAMS

    # -----------------------------------------------------------------------------------------------------------------------------
    # Datasets
    data_name = "wiki2_" + str(cfg.data.n_train) + "_" + str(cfg.data.n_eval)+"_stride"+str(cfg.data.stride)

    def __get_datasets__():
        datasets = lib_data.get_wikitext2_dataset(block_size=cfg.data.seq_len, max_train_samples=cfg.data.n_train,
                                                  max_eval_samples=cfg.data.n_eval, stride=cfg.data.stride)
        return datasets

    # -----------------------------------------------------------------------------------------------------------------------------
    # Architecture

    def __get_arch__():

        model = Transformer(cfg)
        model_name = f"GPT2-mini-pretrained"
        return model, model_name

    # -----------------------------------------------------------------------------------------------------------------------------
    # Optimizer
    def __get_optim__():

        base_string = f"1b{cfg.optim.b1}_2b{cfg.optim.b2}_3b{cfg.optim.b3}_lr{cfg.optim.lr}_warmup{cfg.optim.warmup_steps}"
        if cfg.optim.wd > 0:
            base_string += f"_wd{cfg.optim.wd}"
        if cfg.optim.gn_clip is not None:
            base_string += f"_GNclip{cfg.optim.gn_clip}"
        if (cfg.optim.sam is None) or cfg.optim.sam == '':
            warmup_scheduler = optax.linear_schedule(init_value=0.0, end_value=cfg.optim.lr,
                                                     transition_steps=cfg.optim.warmup_steps,
                                                     transition_begin=0, )
            constant_scheduler = optax.constant_schedule(cfg.optim.lr)
            lr_scheduler = optax.join_schedules([warmup_scheduler, constant_scheduler],
                                                boundaries=[cfg.optim.warmup_steps])
            optimizer = modules.get_sgd_optimizer(lr_scheduler, cfg.optim.b1, cfg.optim.b2, cfg.optim.b3, cfg.optim.wd,
                                                  cfg.optim.gn_clip, verbose=False)
            optim_name = f"sgdFam_{base_string}"

        elif cfg.optim.sam[-3:] == 'sam':
            assert cfg.optim.sam_rho is not None
            adv_lr = cfg.optim.sam_rho * cfg.optim.lr
            warmup_scheduler = optax.linear_schedule(init_value=0.0, end_value=cfg.optim.lr,
                                                     transition_steps=cfg.optim.warmup_steps,
                                                     transition_begin=0, )
            constant_scheduler = optax.constant_schedule(cfg.optim.lr)
            lr_scheduler = optax.join_schedules([warmup_scheduler, constant_scheduler],
                                                boundaries=[cfg.optim.warmup_steps])

            adv_warmup_scheduler = optax.linear_schedule(init_value=0.0, end_value=adv_lr,
                                                         transition_steps=cfg.optim.warmup_steps,
                                                         transition_begin=0, )
            adv_constant_scheduler = optax.constant_schedule(adv_lr)
            adv_lr_scheduler = optax.join_schedules([adv_warmup_scheduler, adv_constant_scheduler],
                                                    boundaries=[cfg.optim.warmup_steps])

            base_opt = modules.get_sgd_optimizer(lr_scheduler, cfg.optim.b1, cfg.optim.b2, cfg.optim.b3, cfg.optim.wd,
                                                 cfg.optim.gn_clip, verbose=False)
            adv_opt = modules.get_sgd_optimizer(adv_lr_scheduler, cfg.optim.b1, cfg.optim.b2, cfg.optim.b3,
                                                cfg.optim.wd, cfg.optim.gn_clip, verbose=False)

            if cfg.optim.sam == 'sam':
                optimizer = sfo.sam(base_opt, adv_opt, sync_period=cfg.optim.sam_sync, opaque_mode=True)  # sam opt
                optim_name = f"sgdFam-SAM_{base_string}_rho{cfg.optim.sam_rho}_syncT{cfg.optim.sam_sync}"
            elif cfg.optim.sam == 'asam':
                optimizer = sfo.asam(base_opt, adv_opt, sync_period=cfg.optim.sam_sync, opaque_mode=True)  # sam opt
                optim_name = f"sgdFam-aSAM_{base_string}_rho{cfg.optim.sam_rho}_syncT{cfg.optim.sam_sync}"
            if cfg.optim.sam == 'lsam':
                optimizer = sfo.looksam(base_opt, adv_opt, sync_period=cfg.optim.sam_sync, beta=0.9,
                                        opaque_mode=True)  # sam opt
                optim_name = f"sgdFam-lSAM_{base_string}_rho{cfg.optim.sam_rho}_syncT{cfg.optim.sam_sync}"

        else:
            raise NotImplementedError

        assert cfg.optim.grad_accum >= 1
        if cfg.optim.grad_accum > 1:
            optim_name += f"_ebs{cfg.optim.bs*cfg.optim.grad_accum}"
        else:
            optim_name += f"_bs{cfg.optim.bs}"

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

    train_loader = lib_data.NumpyLoader(datasets[0], batch_size=cfg.optim.bs, shuffle=True, num_workers=n_workers)
    for sample_batch in train_loader:
        break

    test_loader = lib_data.NumpyLoader(datasets[1], batch_size=cfg.optim.eval_bs, num_workers=n_workers)
    dataloaders = [train_loader, test_loader]

    model, model_name = __get_arch__()
    model_name += "_seed" + str(cfg.optim.seed)

    optim, optim_name = __get_optim__()

    init_rng = jax.random.PRNGKey(cfg.optim.seed)
    state = create_train_state(model, optim, sample_batch[0], init_rng, option=option)
    del init_rng  # Must not be used anymore.

    if cfg.log.demo_outputs: token_predictions(state, sample_batch)
    # load GPT2 weights
    state = load_params_gpt2mini(cfg, state)
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
        experiment_name, lse = utils.find_latest_exp_no_epoch(experiment_name, max_eps=cfg.optim.n_epochs,
                                                     cbs=cb_name_list, verbose=True)
        metrics_history = utils.load_thing("traj/" + experiment_name + "/metrics.pkl")
        print(f"tr_acc: {metrics_history['train_accuracy'][-1]:0%}, te_acc: {metrics_history['test_accuracy'][-1]:0%}")
        metrics_history['lse'] = [lse]
        if cfg.cb.compute_hessian:
            eigvals = utils.load_thing("traj/" + experiment_name + "/eigvals.pkl")
            metrics_history['eigvals'] = eigvals
            print(f"sharp: {metrics_history['eigvals'][-1][0]}")

    except FileNotFoundError:
        metrics_history, state = training.train_model(state, model, cfg.optim.loss_fn, metrics_history, cfg.optim.n_epochs, dataloaders, \
                                               experiment_name, cbs, option=option, force_fb=cfg.optim.force_fb,
                                               tqdm_over_epochs=cfg.log.tqdm_freq, eval_freq=cfg.log.eval_freq, gradient_accumulation=cfg.optim.grad_accum,
                                                      return_state=True)

    if cfg.log.demo_outputs: token_predictions(state, sample_batch)

    out_str += f"tr_acc: {metrics_history['train_accuracy'][-1]:0%}, te_acc: {metrics_history['test_accuracy'][-1]:0%}"
    if cfg.cb.compute_hessian:
        eigvals = utils.load_thing("traj/" + experiment_name + "/eigvals.pkl")
        metrics_history['eigvals'] = eigvals
        out_str += f"sharp: {metrics_history['eigvals'][-1][0]}"

    metrics_history['experiment_name'] = experiment_name

    return out_str, metrics_history, datasets
