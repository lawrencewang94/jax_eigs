import jax
import jax.numpy as jnp
from functools import partial
import treex as tx
# from modules import VGGBlock, ResBlock, AlexNet
from refactor.modules import ResBlock

seed = 0
def id_print(x):
    print(x.shape)
    return x

def get_ResNet3(depth, block_depth=3, n_out=10, base_width=16, multiplier=2, use_BN=True, use_DO=False,
                res_conv='Identity', check_shape=False):
    # legacy: True = Conv, False = Identity
    if isinstance(res_conv, (int)):
        if res_conv:
            res_conv = 'Conv'
        else:
            res_conv = 'Identity'

    assert depth > 0
    model_mode = ""
    model_mode += "_BN" if use_BN else ""
    model_mode += "_DO" if use_DO else ""

    layers_list = []
    # input layer
    layers_list += [
        tx.Conv(base_width, [3, 3], strides=[1, 1], use_bias=False),
        tx.BatchNorm() if use_BN else lambda x: x,
        tx.Dropout(0.1) if use_DO else lambda x: x,
        jax.nn.relu,
    ] # layer 1 and 2

    for i in range(depth):
        width_factor = multiplier ** i
        layers_list += [
            lambda x: id_print(x) if check_shape else x,
            ResBlock(width_factor * base_width, 2, multiplier, dropout=use_DO, bn=use_BN, sc_conv=res_conv),  # 9
            lambda x: id_print(x) if check_shape else x,
            ResBlock(width_factor * base_width, 1, multiplier,  dropout=use_DO, bn=use_BN, sc_conv=res_conv),  # 11
            lambda x: id_print(x) if check_shape else x,
            ResBlock(width_factor * base_width, 1, multiplier, dropout=use_DO, bn=use_BN, sc_conv=res_conv),  # 13
        ]

    # output layer
    layers_list += [
        lambda x: id_print(x) if check_shape else x,
        partial(jnp.mean, axis=(1, 2)),
        lambda x: id_print(x) if check_shape else x,
        tx.Linear(n_out),
    ]
    model = tx.Sequential(*layers_list)
    name_depth = 2+6*depth
    name_conv = "res-" + res_conv[:4]
    model_name = f"ResNet{name_depth:d}_out{n_out:d}_base{base_width:d}_mult{multiplier:d}_{name_conv}_{model_mode}"
    return model, model_name


def get_MLP(depth, n_out, n_h, use_BN=True, use_DO=False, input_flatten=False):
    assert depth > 0
    model_layers = []
    if input_flatten:
        model_layers+= [
            lambda x: x.reshape((x.shape[0], -1)),
        ]

    for i in range(depth):
        model_layers.append(tx.Linear(n_h))
        tx.BatchNorm() if use_BN else lambda x:x,
        tx.Dropout(0.1) if use_DO else lambda x:x,
        model_layers.append(jax.nn.relu)
    model_layers.append(tx.Linear(n_out))
    model = tx.Sequential(*model_layers)

    model_mode = ""
    model_mode += "_BN" if use_BN else ""
    model_mode += "_DO" if use_DO else ""

    model_name = f"mlp{depth:d}_out{n_out:d}_h{n_h:d}_{model_mode}"
    return model, model_name


def get_VGG(depth, block_depth=2, n_out=10, base_width=16, multiplier=2, use_BN=True, use_DO=False, check_shape=False):
    assert block_depth > 0
    assert depth > 0
    model_layers = []
    for i in range(depth):
        width_factor = multiplier ** i
        model_layers += [
            lambda x: id_print(x) if check_shape else x,
            VGGBlock(width_factor * base_width, depth=block_depth, p_drop=0.1, dropout=use_DO, bn=use_BN),
        ]

    model_layers += [
        lambda x: id_print(x) if check_shape else x,
        lambda x: x.reshape((x.shape[0], -1)), # flatten
        lambda x: id_print(x) if check_shape else x,
        tx.Linear(width_factor*base_width), # width of last layer
        tx.BatchNorm() if use_BN else lambda x: x,
        tx.Dropout(0.1) if use_DO else lambda x: x,
        jax.nn.relu,
        tx.Linear(n_out),
    ]
    model = tx.Sequential(*model_layers)

    model_mode = ""
    model_mode += "_BN" if use_BN else ""
    model_mode += "_DO" if use_DO else ""
    name_depth = block_depth * depth + 1

    model_name = f"VGG{name_depth:d}_out{n_out:d}_base{base_width:d}_mult{multiplier:d}_{model_mode}"
    return model, model_name


def get_model(model_arch, use_BN=True, use_DO=False, resnet_base=16, sc_conv=False):
    # resnet_base = 4  # 16 for cifar
    model_mode = ""
    model_mode += "_BN" if use_BN else ""
    model_mode += "_DO" if use_DO else ""
    if model_arch == 'AlexNet':

        model = tx.Sequential(
            AlexNet(64, 64, 256, 128, p_drop=0.2, dropout=use_DO, bn=use_BN),
            tx.Linear(10),
        )
        model_name = f"AlexNet_64_64_256_128" + "_seed" + str(seed) + "_" + model_mode

    elif model_arch == 'VGG1':

        model = tx.Sequential(
            VGGBlock(32, p_drop=0.2, dropout=use_DO, bn=use_BN),
            lambda x: x.reshape((x.shape[0], -1)),
            tx.Linear(32),
            tx.BatchNorm() if use_BN else lambda x: x,
            tx.Dropout(0.2) if use_DO else lambda x: x,
            jax.nn.relu,
            tx.Linear(10),
        )
        model_name = f"VGG1_32_32" + "_seed" + str(seed) + "_" + model_mode

    elif model_arch == 'VGG2':

        model = tx.Sequential(
            VGGBlock(16, p_drop=0.2, dropout=use_DO, bn=use_BN),
            VGGBlock(32, p_drop=0.2, dropout=use_DO, bn=use_BN),
            lambda x: x.reshape((x.shape[0], -1)),
            tx.Linear(32),
            tx.BatchNorm() if use_BN else lambda x: x,
            tx.Dropout(0.2) if use_DO else lambda x: x,
            jax.nn.relu,
            tx.Linear(10),
        )
        model_name = f"VGG2_32_64_32" + "_seed" + str(seed) + "_" + model_mode

    elif model_arch == 'ResNet20':
        #         sc_conv = True
        model = tx.Sequential(
            tx.Conv(resnet_base, [3, 3], strides=[1, 1], use_bias=False),
            tx.BatchNorm() if use_BN else lambda x: x,
            tx.Dropout(0.1) if use_DO else lambda x: x,
            jax.nn.relu,
            ResBlock(resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 3
            ResBlock(resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 5
            ResBlock(resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 7
            ResBlock(2 * resnet_base, 2, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 9
            ResBlock(2 * resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 11
            ResBlock(2 * resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 13
            ResBlock(4 * resnet_base, 2, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 15
            ResBlock(4 * resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 17
            ResBlock(4 * resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 19
            partial(jnp.mean, axis=(1, 2)),
            tx.Linear(10),
        )
        model_name = f"ResNet20base" + str(resnet_base) + "_seed" + str(seed) + "_" + model_mode
        if sc_conv:
            model_name += "_B"
        else:
            model_name += "_A"


    elif model_arch == 'ResNet14':
        #         sc_conv = True
        model = tx.Sequential(
            tx.Conv(resnet_base, [3, 3], strides=[1, 1], use_bias=False),
            tx.BatchNorm() if use_BN else lambda x: x,
            tx.Dropout(0.1) if use_DO else lambda x: x,
            jax.nn.relu,
            ResBlock(resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 3
            ResBlock(resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 5
            ResBlock(resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 7
            ResBlock(2 * resnet_base, 2, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 9
            ResBlock(2 * resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 11
            ResBlock(2 * resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 13
            partial(jnp.mean, axis=(1, 2)),
            tx.Linear(10),
        )
        model_name = f"ResNet14base" + str(resnet_base) + "_seed" + str(seed) + "_" + model_mode
        if sc_conv == True:
            model_name += "_B"
        else:
            model_name += "_A"

    elif model_arch == 'ResNet8':
        model = tx.Sequential(
            tx.Conv(resnet_base, [3, 3], strides=[1, 1], use_bias=False),
            tx.BatchNorm() if use_BN else lambda x: x,
            tx.Dropout(0.1) if use_DO else lambda x: x,
            jax.nn.relu,
            ResBlock(resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 3
            ResBlock(resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 5
            ResBlock(resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 7
            partial(jnp.mean, axis=(1, 2)),
            tx.Linear(10),
        )
        model_name = f"ResNet8base" + str(resnet_base) + "_seed" + str(seed) + "_" + model_mode
        if sc_conv == True:
            model_name += "_B"
        else:
            model_name += "_A"

    elif model_arch == 'ResNet26':
        model = tx.Sequential(
            tx.Conv(resnet_base, [3, 3], strides=[1, 1], use_bias=False),
            tx.BatchNorm() if use_BN else lambda x: x,
            tx.Dropout(0.1) if use_DO else lambda x: x,
            jax.nn.relu,
            ResBlock(resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 3
            ResBlock(resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 5
            ResBlock(resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 7
            ResBlock(2 * resnet_base, 2, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 9
            ResBlock(2 * resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 11
            ResBlock(2 * resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 13
            ResBlock(4 * resnet_base, 2, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 15
            ResBlock(4 * resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 17
            ResBlock(4 * resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 19
            ResBlock(8 * resnet_base, 2, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 21
            ResBlock(8 * resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 23
            ResBlock(8 * resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 25
            partial(jnp.mean, axis=(1, 2)),
            tx.Linear(10),
        )
        model_name = f"ResNet26base" + str(resnet_base) + "_seed" + str(seed) + "_" + model_mode
        if sc_conv == True:
            model_name += "_B"
        else:
            model_name += "_A"

    elif model_arch == 'ResNet32':
        model = tx.Sequential(
            tx.Conv(resnet_base, [3, 3], strides=[1, 1], use_bias=False),
            tx.BatchNorm() if use_BN else lambda x: x,
            tx.Dropout(0.1) if use_DO else lambda x: x,
            jax.nn.relu,
            ResBlock(resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 3
            ResBlock(resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 5
            ResBlock(resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 7
            ResBlock(2 * resnet_base, 2, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 9
            ResBlock(2 * resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 11
            ResBlock(2 * resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 13
            ResBlock(4 * resnet_base, 2, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 15
            ResBlock(4 * resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 17
            ResBlock(4 * resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 19
            ResBlock(8 * resnet_base, 2, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 21
            ResBlock(8 * resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 23
            ResBlock(8 * resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 25
            ResBlock(16 * resnet_base, 2, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 27
            ResBlock(16 * resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 29
            ResBlock(16 * resnet_base, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 31
            partial(jnp.mean, axis=(1, 2)),
            tx.Linear(10),
        )
        model_name = f"ResNet32base" + str(resnet_base) + "_seed" + str(seed) + "_" + model_mode
        if sc_conv == True:
            model_name += "_B"
        else:
            model_name += "_A"


    elif model_arch == 'ResNet18':
        sc_conv = False
        model = tx.Sequential(
            tx.Conv(64, [3, 3], strides=[1, 1], use_bias=False),
            tx.BatchNorm() if use_BN else lambda x: x,
            tx.Dropout(0.1) if use_DO else lambda x: x,
            jax.nn.relu,
            ResBlock(64, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 3
            ResBlock(64, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 5
            ResBlock(128, 2, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 7
            ResBlock(128, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 9
            ResBlock(256, 2, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 11
            ResBlock(256, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 13
            ResBlock(512, 2, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 15
            ResBlock(512, 1, dropout=use_DO, bn=use_BN, sc_conv=sc_conv),  # 17
            partial(jnp.mean, axis=(1, 2)),
            tx.Linear(10),
        )
        model_name = f"ResNet18" + "_seed" + str(seed) + "_" + model_mode
        if sc_conv:
            model_name += "_B"
        else:
            model_name += "_A"

    elif model_arch == 'ResNet9':

        model = tx.Sequential(
            ResNet9(10, dropout=use_DO, bn=use_BN),
        )
        model_name = f"ResNet9" + "_seed" + str(seed) + "_" + model_mode
    return model, model_name


class ResNet20(nn.Module):
    """A ResNet20 model."""
    resnet_base = 8
    use_DO = False
    use_BN = True
    sc_conv = 'Identity'
    deterministic: tp.Optional[bool] = None

    @nn.compact
    def __call__(self, x, train=True):
        deterministic = not train
        deterministic = nn.merge_param('deterministic', self.deterministic, deterministic)
        x = nn.Conv(self.resnet_base, [3, 3], strides=[1, 1], use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=deterministic)(x) if self.use_BN else modules.Lambda(f=lambda x: x)(x)
        x = nn.Dropout(0.1)(x, deterministic=deterministic) if self.use_DO else modules.Lambda(f=lambda x: x)(x)
        x = modules.Lambda(jax.nn.relu)(x)

        x = modules.ResBlock(out_channels=1 * self.resnet_base, dropout=self.use_DO, bn=self.use_BN,
                             sc_conv=self.sc_conv)(x, deterministic)  # 3
        x = modules.ResBlock(out_channels=1 * self.resnet_base, dropout=self.use_DO, bn=self.use_BN,
                             sc_conv=self.sc_conv)(x, deterministic)  # 5
        x = modules.ResBlock(out_channels=1 * self.resnet_base, dropout=self.use_DO, bn=self.use_BN,
                             sc_conv=self.sc_conv)(x, deterministic)  # 7

        x = modules.ResBlock(out_channels=2 * self.resnet_base, strides=2, dropout=self.use_DO, bn=self.use_BN,
                             sc_conv=self.sc_conv)(x, deterministic)  # 9
        x = modules.ResBlock(out_channels=2 * self.resnet_base, dropout=self.use_DO, bn=self.use_BN,
                             sc_conv=self.sc_conv)(x, deterministic)  # 11
        x = modules.ResBlock(out_channels=2 * self.resnet_base, dropout=self.use_DO, bn=self.use_BN,
                             sc_conv=self.sc_conv)(x, deterministic)  # 13

        x = modules.ResBlock(out_channels=4 * self.resnet_base, strides=2, dropout=self.use_DO, bn=self.use_BN,
                             sc_conv=self.sc_conv)(x, deterministic)  # 15
        x = modules.ResBlock(out_channels=4 * self.resnet_base, dropout=self.use_DO, bn=self.use_BN,
                             sc_conv=self.sc_conv)(x, deterministic)  # 17
        x = modules.ResBlock(out_channels=4 * self.resnet_base, dropout=self.use_DO, bn=self.use_BN,
                             sc_conv=self.sc_conv)(x, deterministic)  # 19

        x = partial(jnp.mean, axis=(1, 2))(x)
        # x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(10)(x)

        return x