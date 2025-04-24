from torchvision import datasets, transforms
import torchvision
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

import torch
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

data_dir = "data/"


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        # print("collate option A")
        return np.stack(batch)
    elif isinstance(batch[0], torch.Tensor):
        # print("collate option B")
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        # print("collate option C")
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        # print("collate option D")
        return np.array(batch)


class NumpyLoader(DataLoader):
    def __init__(self, dataset, batch_size=1,
                 shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0,
                 pin_memory=False, drop_last=False,
                 timeout=0, worker_init_fn=None,ood_name=""):
        super(self.__class__, self).__init__(dataset,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             sampler=sampler,
                                             batch_sampler=batch_sampler,
                                             num_workers=num_workers,
                                             collate_fn=numpy_collate,
                                             pin_memory=pin_memory,
                                             drop_last=drop_last,
                                             timeout=timeout,
                                             worker_init_fn=worker_init_fn)
        self.name=ood_name


class Dataset(torch.utils.data.Dataset):
    #Characterizes a dataset for PyTorch
    def __init__(self, xs, ys):
        self.data = xs
        self.targets = ys
        self.len = len(xs)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # Load data and get label
        X = self.data[index]
        y = self.targets[index]

        return X, y


class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))


class CastNP(object):
    def __call__(self, pic):
        a = np.array(pic, dtype=jnp.float32)
        return a


class CastNP_MNIST(object):
    def __call__(self, pic):
        a = np.array(pic, dtype=jnp.float32)
        a = np.moveaxis(a, [0], [2])
        return a


def get_mnist(flatten=False, tr_indices=60000, te_indices=10000, hess_indices=1000, tr_classes=10, te_classes=10, hess_classes=10,
              visualise=True, rand_start=False):
    normalize = transforms.Normalize((0.1307,), (0.3081,))  # MNIST
    if flatten:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            FlattenAndCast(),
        ])

    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            CastNP_MNIST(),
        ])
    train_dataset = datasets.MNIST(root=data_dir, train=True,
                download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False,
                download=True, transform=transform)
    hess_dataset = datasets.MNIST(root=data_dir, train=True,
                download=True, transform=transform)
    # print(train_dataset[0])

    if hess_indices is not None:
        assert hess_classes is not None
        hv_idx = np.where(np.array(train_dataset.targets) < hess_classes)[0]
        if hess_indices < 60000:
            if not rand_start:
                hv_idx = hv_idx[torch.arange(hess_indices)]
            else:
                hv_idx = hv_idx[torch.arange(hess_indices) + torch.randint(high=len(hess_dataset) - hess_indices - 1, size=(1,))]
        hess_inputs = np.array([hess_dataset.__getitem__(i)[0] for i in hv_idx])
        hess_targets = np.array([hess_dataset.__getitem__(i)[1] for i in hv_idx])
        hess_dataset = Dataset(hess_inputs, hess_targets)

    tr_idx = np.where(train_dataset.targets < tr_classes)[0]
    # print(idx)
    train_dataset = data_utils.Subset(train_dataset, tr_idx)

    te_idx = np.where(test_dataset.targets < te_classes)[0]
    # print(len(idx[0]))
    test_dataset = data_utils.Subset(test_dataset, te_idx)

    if tr_indices < 60000:
        if not rand_start:
            indices = torch.arange(tr_indices)
        else:
            indices = torch.arange(tr_indices) + torch.randint(high=len(train_dataset) - tr_indices - 1, size=(1,))
        train_dataset = data_utils.Subset(train_dataset, indices)
    if te_indices < 10000:
        if not rand_start:
            indices = torch.arange(te_indices)
        else:
            indices = torch.arange(te_indices) + torch.randint(high=len(test_dataset) - te_indices - 1, size=(1,))
        test_dataset = data_utils.Subset(test_dataset, indices)

    if visualise:
        raw_dataset = datasets.MNIST(root=data_dir, train=True,
                                            download=True, transform=transforms.ToTensor())
        imgs = []
        lbls = []
        for i in range(10):
            idx = tr_idx[i]
            imgs.append(np.array(raw_dataset.data[idx]) / 255)
            lbls.append(raw_dataset.targets[idx])
        # imgs, lbls = sample
        imgs = np.array(imgs)
        lbls = np.array(lbls)
        imgs = imgs.reshape(imgs.shape[0], 1, 28, 28)
        plt.figure(figsize=(15, 10))
        grid = torchvision.utils.make_grid(nrow=20, tensor=torch.Tensor(imgs))
        # print(f"image tensor: {imgs.shape}")
        print(f"class labels: {lbls}")
        plt.imshow(np.transpose(grid, axes=(1, 2, 0)), cmap='gray')
        plt.show()

    return train_dataset, test_dataset, hess_dataset


def visualise_mnist(sample, n=10):
    imgs, lbls = sample
    imgs = imgs.reshape(imgs.shape[0], 1, 28, 28)
    imgs = imgs[:n]
    lbls = lbls[:n]
    plt.figure(figsize=(15, 10))
    grid = torchvision.utils.make_grid(nrow=20, tensor=torch.Tensor(imgs))
    # print(f"image tensor: {imgs.shape}")
    print(f"class labels: {lbls}")
    plt.imshow(np.transpose(grid, axes=(1, 2, 0)), cmap='gray')
    plt.show()


def get_cla_dataset(data_func, n_train, n_eval, noise_std=1, train_range=[-4, 4], val_range=[-4, 4], ood_range=[-6, 6],
                    y_range=None,
                    show=False, show_all=False, sorted_val=False):
    if y_range is None:
        tmp_x = np.linspace(train_range[0], train_range[1], 100)
        tmp_y = data_func(tmp_x)
        y_range = (np.min(tmp_y), np.max(tmp_y))
    else:
        assert len(y_range) == 2

    # noise_std = 3
    # train_range, val_range, test_range = utils.get_data_ranges()
    # utils.set_seed(0)

    def draw_examples(n, x_range):
        x = np.random.uniform(x_range[0], x_range[1], n)
        y = np.random.uniform(y_range[0], y_range[1], n)
        eps = np.random.normal(0, noise_std, n)
        #     y = np.power(x,3) + eps
        f_x = data_func(x) + eps
        y_labels = np.array([1 if y[i] > f_x[i] else 0 for i in range(len(y))])
        #         y_labels = y_labels[:, np.newaxis]
        inputs = np.concatenate([x[:, np.newaxis], y[:, np.newaxis]], axis=1)

        return (inputs, y_labels)

    toy_dataset = draw_examples(n_train, train_range)
    inputs, labels = toy_dataset
    val_dataset = draw_examples(n_train, val_range)
    val_inputs, val_labels = val_dataset
    ood_dataset = draw_examples(n_train, ood_range)
    ood_inputs, ood_labels = ood_dataset

    if show:
        xs = np.linspace(train_range[0], train_range[1], n_eval)
        # ys = np.power(xs,3)
        ys = data_func(xs)
        plt.plot(xs, ys, "k")
        pos_inds = np.where(toy_dataset[1] == 1)[0]
        neg_inds = np.where(toy_dataset[1] == 0)[0]

        plt.scatter(toy_dataset[0][pos_inds, 0], toy_dataset[0][pos_inds, 1], color='r', label='Positive Samples')
        plt.scatter(toy_dataset[0][neg_inds, 0], toy_dataset[0][neg_inds, 1], color='b', label='Negative Samples')

        plt.legend(loc='best')
        plt.ylabel("y")
        plt.xlabel("x")
        plt.title("Toy Dataset - shows classification boundary as a function")
        plt.show()

    tr_dataset = Dataset(torch.tensor(inputs, dtype=torch.float32),
                            torch.tensor(labels, dtype=torch.int))

    val_dataset = Dataset(torch.tensor(val_inputs, dtype=torch.float32),
                             torch.tensor(val_labels, dtype=torch.int))

    ood_dataset = Dataset(torch.tensor(ood_inputs, dtype=torch.float32),
                             torch.tensor(ood_labels, dtype=torch.int))

    return tr_dataset, val_dataset, ood_dataset


def get_reg_dataset(data_func, n_train, n_eval, noise_std=1, train_range=[-4, 4], val_range=[-4, 4], ood_range=[-6, 6],
                show=False, show_all=False, sorted_val=False):

    # noise_std = 3
    # train_range, val_range, test_range = utils.get_data_ranges()
    # utils.set_seed(0)

    def draw_examples(n, x_range, sorted=False):
        x = np.random.uniform(x_range[0], x_range[1], n)
        if sorted:
            x = np.sort(x)
        eps = np.random.normal(0, noise_std, n)
        #     y = np.power(x,3) + eps
        y = data_func(x) + eps
        return (x, y)

    toy_dataset = draw_examples(n_train, train_range)
    inputs, labels = toy_dataset
    val_dataset = draw_examples(n_train, val_range, sorted=sorted_val)
    val_inputs, val_labels = val_dataset
    ood_dataset = draw_examples(n_train, ood_range, sorted=sorted_val)
    ood_inputs, ood_labels = ood_dataset

    if show:
        xs = np.linspace(train_range[0], train_range[1], n_eval)
        # ys = np.power(xs,3)
        ys = data_func(xs)
        plt.plot(xs, ys, "k")
        plt.scatter(toy_dataset[0], toy_dataset[1], color='r', label='Training Samples')
        plt.legend(loc='best')
        plt.ylabel("y")
        plt.xlabel("x")
        plt.title("Toy Dataset")
        plt.show()

    if show_all:
        xs = np.linspace(train_range[0], train_range[1], n_eval)
        # ys = np.power(xs,3)
        ys = data_func(xs)
        plt.plot(xs, ys, "k")
        plt.scatter(val_dataset[0], val_dataset[1], color='r', label='Val Samples')
        plt.legend(loc='best')
        plt.ylabel("y")
        plt.xlabel("x")
        plt.title("Val Dataset")
        plt.show()

        xs = np.linspace(train_range[0], train_range[1], n_eval)
        # ys = np.power(xs,3)
        ys = data_func(xs)
        plt.plot(xs, ys, "k")
        plt.scatter(ood_dataset[0], ood_dataset[1], color='r', label='ood Samples')
        plt.legend(loc='best')
        plt.ylabel("y")
        plt.xlabel("x")
        plt.title("ood Dataset")
        plt.show()

    tr_dataset = Dataset(torch.tensor(inputs, dtype=torch.float32)[:, None],
                      torch.tensor(labels, dtype=torch.float32)[:, None])

    val_dataset = Dataset(torch.tensor(val_inputs, dtype=torch.float32)[:, None],
                          torch.tensor(val_labels, dtype=torch.float32)[:, None])

    ood_dataset = Dataset(torch.tensor(ood_inputs, dtype=torch.float32)[:, None],
                          torch.tensor(ood_labels, dtype=torch.float32)[:, None])

    return tr_dataset, val_dataset, ood_dataset


def swiss_roll_2d(l_phi, sigma, m, gap=1, noise_type="bal", sorted=False):
    if noise_type == 'bal':
        xi = np.random.rand(m) - 0.5
    elif noise_type == 'pos':
        xi = np.random.rand(m)
    elif noise_type == 'neg':
        xi = np.random.rand(m) - 1

    phi = l_phi * np.random.rand(m) + gap
    if sorted:
        phi = np.sort(phi)

    phi_n = phi + sigma * xi

    X = 1. * (phi_n) * np.sin(phi)
    Y = 1. * (phi_n) * np.cos(phi)

    return np.array([X, Y]).transpose()


def make_swiss_rolls(l_phi, s1, s2, m, gap=1, roll_type="bal", sorted=False):

    if roll_type == "food":
        sr0 = swiss_roll_2d(l_phi - np.pi / 4, s1, m, gap, noise_type='bal', sorted=sorted)
        sr1 = -swiss_roll_2d(l_phi + 3 * np.pi / 4, s2, m, gap, noise_type='bal', sorted=sorted)
    elif roll_type == "bal":
        sr0 = swiss_roll_2d(l_phi, s1, m, gap, noise_type='bal', sorted=sorted)
        sr1 = -swiss_roll_2d(l_phi, s2, m, gap, noise_type='bal', sorted=sorted)
    elif roll_type == "close":
        sr0 = swiss_roll_2d(l_phi, s1, m, gap, noise_type='pos', sorted=sorted)
        sr1 = -swiss_roll_2d(l_phi, s2, m, gap, noise_type='neg', sorted=sorted)
    elif roll_type == "far":
        sr0 = swiss_roll_2d(l_phi, s1, m, gap, noise_type='neg', sorted=sorted)
        sr1 = -swiss_roll_2d(l_phi, s2, m, gap, noise_type='pos', sorted=sorted)

    return sr0, sr1


def get_sr_cla(noise, n_tr, n_ev, tr_l=10, ood_l=15, gap=1, show=False, sorted_val=False):
    assert n_tr % 2 == 0
    assert n_ev % 2 == 0
    n_tr = int(n_tr / 2)
    n_ev = int(n_ev / 2)

    sr0, sr1 = make_swiss_rolls(tr_l, noise, noise, n_tr, gap=gap, roll_type="bal")

    if show:
        plt.scatter(sr0[:, 0], sr0[:, 1], c="r")
        plt.scatter(sr1[:, 0], sr1[:, 1], c="b")

    tr_dataset = Dataset(torch.cat([torch.tensor(sr0, dtype=torch.float32),
                                               torch.tensor(sr1, dtype=torch.float32)], dim=0),
                                    torch.cat([torch.tensor([0]).repeat(n_tr),
                                               torch.tensor([1]).repeat(n_tr)], dim=0))

    sr0, sr1 = make_swiss_rolls(tr_l, noise, noise, n_ev, gap=gap, roll_type="bal", sorted=sorted_val)

    te_dataset = Dataset(torch.cat([torch.tensor(sr0, dtype=torch.float32),
                                               torch.tensor(sr1, dtype=torch.float32)], dim=0),
                                    torch.cat([torch.tensor([0]).repeat(n_ev),
                                               torch.tensor([1]).repeat(n_ev)], dim=0))

    sr0, sr1 = make_swiss_rolls(ood_l, noise, noise, n_ev, gap=tr_l+gap, roll_type="bal", sorted=sorted_val)

    ood_dataset = Dataset(torch.cat([torch.tensor(sr0, dtype=torch.float32),
                                                torch.tensor(sr1, dtype=torch.float32)], dim=0),
                                     torch.cat([torch.tensor([0]).repeat(n_ev),
                                                torch.tensor([1]).repeat(n_ev)], dim=0))

    if show:
        plt.scatter(sr0[:, 0], sr0[:, 1], c="r", marker="x")
        plt.scatter(sr1[:, 0], sr1[:, 1], c="b", marker="x")
        plt.show()

    return tr_dataset, te_dataset, ood_dataset


def make_chocolate_roll():
    # :)
    sr0, sr1 = make_swiss_rolls(10, 1, 1, 1000, roll_type="food")
    plt.scatter(sr0[:, 0], sr0[:, 1], c="moccasin")
    plt.scatter(sr1[:, 0], sr1[:, 1], c="saddlebrown")


# functions for data



def make_sine_shift(bs, n_c, n_t, x_domain=[-10, 10], shift_range=[-10, 10], freq=1, noise=1):
    # each row in a batch is a different function
    # fixed n_c, n_t for now

    x = np.random.uniform(x_domain[0], x_domain[1], size=(bs, n_c + n_t, 1))
    shift = np.repeat(np.random.uniform(shift_range[0], shift_range[1], size=(bs, 1, 1)), repeats=n_c + n_t, axis=1)
    y = np.sin(freq * x) + shift + np.random.normal(loc=0, scale=noise)

    c_x = x[:, :n_c, :]
    c_y = y[:, :n_c, :]
    t_x = x[:, n_c:, :]
    t_y = y[:, n_c:, :]

    return c_x, c_y, t_x, t_y


def get_sine_shift(n_train, n_eval, n_c, n_t, x_domain=[-10, 10], shift_range=[-10, 10], freq=1, noise=1, show=False):
    cx, cy, tx, ty = make_sine_shift(n_train, n_c, n_t, x_domain=x_domain, freq=freq, noise=noise)

    if show:
        plt.scatter(cx[0], cy[0], label="context train")
        plt.scatter(tx[0], ty[0], label="target train")
        plt.show()

    train_ds = Dataset(cx, cy, tx, ty)
    cx, cy, tx, ty = make_sine_shift(n_eval, n_c, n_t, x_domain=x_domain, freq=freq, noise=noise)

    if show:
        plt.scatter(cx[0], cy[0], label="context valid")
        plt.scatter(tx[0], ty[0], label="target valid")
        plt.show()

    val_ds = Dataset(cx, cy, tx, ty)
    cx, cy, tx, ty = make_sine_shift(n_eval, n_c, n_t, x_domain=x_domain, freq=freq, noise=noise)
    ood_ds = Dataset(cx, cy, tx, ty)

    return [train_ds, val_ds, ood_ds]


class staticCIFAR10(torch.utils.data.Dataset):
    def __init__(self, orig_ds, max_class=10, expansion_size=1, seed=0):
        super(staticCIFAR10, self).__init__()
        self.orig_ds = orig_ds
        self.max_class = max_class
        self.expansion_size = expansion_size
        np.random.seed(seed)
        self.flip_rng = np.random.choice(2, size=self.__len__())
        self.crop_rng = np.random.choice(8, size=(self.__len__(), 2))
        assert isinstance(expansion_size, int) and expansion_size > 0

    def __getitem__(self, index):

        x, y = self.orig_ds[index//self.expansion_size]  # get the original item
        if self.flip_rng[index] == 1:
            x = x[:, ::-1]
        # random horizontal flip
        # padding = 4,
        padded_x = np.zeros((40, 40, 3))
        padded_x[4:-4, 4:-4, :] = x.copy()
        cropped_x = padded_x[self.crop_rng[index][0]:self.crop_rng[index][0]+32,
                    self.crop_rng[index][1]:self.crop_rng[index][1]+32]
        # my_x = # change input digit image x ?
        # my_y = np.array(torch.randint(0, self.max_class, (1,)).item())# change the original label y ?
        # my_y = 999# change the original label y ?

        return cropped_x, y

    def __len__(self):
        return int(self.orig_ds.__len__()*self.expansion_size)


def get_cifar10(flatten=False, tr_indices=60000, te_indices=10000, tr_classes=10, te_classes=10,
                hess_indices=None, hess_classes=None, visualise=True, rand_start=False,
                one_hot=False, augmentations=False, aug_factor=1, seed=0):
    assert tr_indices // tr_classes <= 6000
    assert te_indices // te_classes <= 1000

    normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))  # CIFAR https://github.com/kuangliu/pytorch-cifar/issues/19
    # normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # CIFAR-pretrain, https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/trainer.py
    if isinstance(augmentations, int):
        assert augmentations == 1 or augmentations == 0
        if augmentations == 0:
            augmentations = 'none'
        else:
            augmentations = 'random'

    transform_list = []
    if augmentations == 'random': # CIFAR 10 Resnet paper
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.RandomCrop(32, 4))

    transform_list.append(transforms.ToTensor())
    transform_list.append(normalize)

    if flatten:
        transform_list.append(FlattenAndCast())
    else:
        transform_list.append(transforms.Lambda(lambda x: torch.moveaxis(x, 0, 2)))
        transform_list.append(CastNP())

    train_transform = transforms.Compose(transform_list)
    if augmentations == 'random':
        test_trainsform = transforms.Compose(transform_list[2:]) # skipping 2 for augments
    else:
        test_trainsform = transforms.Compose(transform_list)

    train_dataset = datasets.CIFAR10(root=data_dir, train=True,
                download=True, transform=train_transform)
    hess_dataset = datasets.CIFAR10(root=data_dir, train=True,
                download=True, transform=test_trainsform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False,
                download=True, transform=test_trainsform)

    if hess_indices is not None:
        assert hess_classes is not None
        hv_idx = np.where(np.array(train_dataset.targets) < hess_classes)[0]
        if hess_indices < 50000:
            if not rand_start:
                hv_idx = hv_idx[torch.arange(hess_indices)]
            else:
                hv_idx = hv_idx[torch.arange(hess_indices) + torch.randint(high=len(hess_dataset) - hess_indices - 1, size=(1,))]
        hess_inputs = np.array([hess_dataset.__getitem__(i)[0] for i in hv_idx])
        hess_targets = np.array([hess_dataset.__getitem__(i)[1] for i in hv_idx])
        if not one_hot:
            hess_dataset = Dataset(hess_inputs, hess_targets)
        else:
            hess_targets_oh = np.array(torch.nn.functional.one_hot(torch.tensor(hess_targets), num_classes=hess_classes))  # change the original label y ?
            hess_dataset = Dataset(hess_inputs, hess_targets_oh)
    else:
        pass
    tr_idx = np.where(np.array(train_dataset.targets) < tr_classes)[0]
    te_idx = np.where(np.array(test_dataset.targets) < te_classes)[0]

    if one_hot:
        train_dataset = data_utils.Subset(onehot_DS(train_dataset, tr_classes), tr_idx)
        test_dataset = data_utils.Subset(onehot_DS(test_dataset, te_classes), te_idx)

    else:
        train_dataset = data_utils.Subset(train_dataset, tr_idx)
        test_dataset = data_utils.Subset(test_dataset, te_idx)

    # tr_idx = np.where(np.array(train_dataset.targets) < tr_classes)[0]
    # # print(idx)
    # train_dataset = data_utils.Subset(train_dataset, tr_idx)
    #
    # te_idx = np.where(np.array(test_dataset.targets) < te_classes)[0]
    # # print(len(idx[0]))
    # test_dataset = data_utils.Subset(test_dataset, te_idx)

    if tr_indices < 50000:
        if not rand_start:
            indices = torch.arange(tr_indices)
        else:
            indices = torch.arange(tr_indices) + torch.randint(high=len(train_dataset)-tr_indices-1, size=(1,))
        train_dataset = data_utils.Subset(train_dataset, indices)

    if te_indices < 10000:
        if not rand_start:
            indices = torch.arange(te_indices)
        else:
            indices = torch.arange(te_indices) + torch.randint(high=len(test_dataset)-te_indices-1, size=(1,))
        test_dataset = data_utils.Subset(test_dataset, indices)

    if augmentations == 'static':
        train_dataset = staticCIFAR10(train_dataset, tr_classes, aug_factor, seed)

    if visualise:
        raw_dataset = datasets.CIFAR10(root=data_dir, train=True,
                                         download=True, transform=transforms.ToTensor())
        rows, columns = 2, 10
        fig = plt.figure(figsize=(10, 3))
        # visualize these random images
        for i in range(1, columns * rows + 1):
            fig.add_subplot(rows, columns, i)
            idx = tr_idx[i-1]
            img = raw_dataset.data[idx]
            lbl = raw_dataset.targets[idx]
            # img = img.reshape(3, 32, 32).transpose(1, 2, 0)
            # print(np.dtype(img[0, 0, 0]), np.max(img), np.min(img))
            plt.imshow(img, vmax=255)
            plt.xticks([])
            plt.yticks([])
            plt.title("{}".format(lbl))
        plt.show()

    return train_dataset, test_dataset, hess_dataset


def visualise_cifar(sample, n=9):
    imgs, lbls = sample
    imgs = imgs.reshape(imgs.shape[0], 3, 32, 32).transpose(0, 2, 3, 1) # CIFAR
    imgs = imgs[:n]
    lbls = lbls[:n]
    rows, columns = 3, 3
    fig = plt.figure(figsize=(4, 4))
    # visualize these random images
    for i in range(1, columns * rows + 1):
        fig.add_subplot(rows, columns, i)
        print(np.dtype(imgs[i-1][0,0,0]), np.max(imgs[i-1]), np.min(imgs[i-1]))
        plt.imshow(imgs[i - 1])
        plt.xticks([])
        plt.yticks([])
        plt.title("{}".format(lbls[i-1]))
    plt.show()


class CorruptedFMNIST(torch.utils.data.Dataset):
    def __init__(self, orig_mnist, max_class=10,):
        super(CorruptedFMNIST, self).__init__()
        self.orig_mnist = orig_mnist
        self.max_class = max_class

    def __getitem__(self, index):
        x, y = self.orig_mnist[index]  # get the original item
        # my_x = # change input digit image x ?
        my_y = np.array(torch.randint(0, self.max_class, (1,)).item())# change the original label y ?
        # my_y = 999# change the original label y ?

        return x, my_y

    def __len__(self):
        return self.orig_mnist.__len__()


class onehot_DS(torch.utils.data.Dataset):
    def __init__(self, orig_mnist, max_class=10,):
        super(onehot_DS, self).__init__()
        self.orig_ds = orig_mnist
        self.max_class = max_class

    def __getitem__(self, index):
        x, y = self.orig_ds[index]  # get the original item
        # my_x = # change input digit image x ?
        my_y = np.array(torch.nn.functional.one_hot(torch.tensor(y), num_classes=self.max_class))# change the original label y ?
        # my_y = 999# change the original label y ?
        return x, my_y

    def __len__(self):
        return self.orig_ds.__len__()


def get_fashion_mnist(flatten=False, tr_indices=60000, te_indices=10000, hess_indices=None,
                      tr_classes=10, te_classes=10, hess_classes=None, corrupt_p=0., one_hot=False, augmentations=False,
                      visualise=True, rand_start=False):

    assert tr_indices // tr_classes <= 6000
    assert te_indices // te_classes <= 1000

    normalize = transforms.Normalize((0.2859,), (0.3530,))  # fMNIST
    if flatten:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            FlattenAndCast(),
        ])

    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            CastNP_MNIST(),
        ])

    train_dataset = datasets.FashionMNIST(root=data_dir, train=True,
                download=True, transform=transform)
    hess_dataset = datasets.FashionMNIST(root=data_dir, train=True,
                download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root=data_dir, train=False,
                download=True, transform=transform)

    if hess_indices is not None:
        assert hess_classes is not None
        hv_idx = np.where(np.array(train_dataset.targets) < hess_classes)[0]
        if hess_indices < 50000:
            if not rand_start:
                hv_idx = hv_idx[torch.arange(hess_indices)]
            else:
                hv_idx = hv_idx[torch.arange(hess_indices) + torch.randint(high=len(hess_dataset) - hess_indices - 1, size=(1,))]
        hess_inputs = np.array([hess_dataset.__getitem__(i)[0] for i in hv_idx])
        hess_targets = np.array([hess_dataset.__getitem__(i)[1] for i in hv_idx])
        if not one_hot:
            hess_dataset = Dataset(hess_inputs, hess_targets)
        else:
            hess_targets_oh = np.array(torch.nn.functional.one_hot(torch.tensor(hess_targets), num_classes=hess_classes))  # change the original label y ?
            hess_dataset = Dataset(hess_inputs, hess_targets_oh)
    else:
        pass

    tr_idx = np.where(train_dataset.targets < tr_classes)[0]
    te_idx = np.where(test_dataset.targets < te_classes)[0]

    # print(idx)
    if one_hot:
        corrupt_set = onehot_DS(CorruptedFMNIST(train_dataset, tr_classes), tr_classes)
        train_dataset = data_utils.Subset(onehot_DS(train_dataset, tr_classes), tr_idx)
        train_dataset_corrupt = data_utils.Subset(corrupt_set, tr_idx)
        test_dataset = data_utils.Subset(onehot_DS(test_dataset, te_classes), te_idx)

    else:
        corrupt_set = CorruptedFMNIST(train_dataset, tr_classes)
        train_dataset = data_utils.Subset(train_dataset, tr_idx)
        train_dataset_corrupt = data_utils.Subset(corrupt_set, tr_idx)
        test_dataset = data_utils.Subset(test_dataset, te_idx)

    # print(len(idx[0]))

    if tr_indices < 60000:
        if not rand_start:
            indices = torch.arange(tr_indices)
        else:
            indices = torch.arange(tr_indices) + torch.randint(high=len(train_dataset)-tr_indices-1, size=(1,))

        rand_indices = indices[torch.randperm(indices.shape[0])]

        indices_clean, indices_corrupt = torch.split(rand_indices, [int(indices.shape[0]*(1-corrupt_p)), int(indices.shape[0]*(corrupt_p))], dim=0)
        if visualise: print("Clean", len(indices_clean), "Corrupt", len(indices_corrupt))
        subset_clean = data_utils.Subset(train_dataset, indices_clean)
        subset_corrupt = data_utils.Subset(train_dataset_corrupt, indices_corrupt)
        # train_dataset = data_utils.Subset(train_dataset, indices)
        train_dataset = data_utils.ConcatDataset([subset_clean, subset_corrupt])

    if te_indices < 10000:
        if not rand_start:
            indices = torch.arange(te_indices)
        else:
            indices = torch.arange(te_indices) + torch.randint(high=len(test_dataset)-te_indices-1, size=(1,))
        test_dataset = data_utils.Subset(test_dataset, indices)

    if visualise:
        raw_dataset = datasets.FashionMNIST(root=data_dir, train=True,
                download=True, transform=transforms.ToTensor())
        imgs = []
        lbls = []
        for i in range(10):
            idx = tr_idx[i]
            imgs.append(np.array(raw_dataset.data[idx])/255)
            # lbls.append(raw_dataset.targets[idx])
            if one_hot:
                lbls.append(np.argmax(corrupt_set[idx][1]))
            else:
                lbls.append(corrupt_set[idx][1])
        # imgs, lbls = sample
        imgs = np.array(imgs)
        lbls = np.array(lbls)
        imgs = imgs.reshape(imgs.shape[0], 1, 28, 28)
        plt.figure(figsize=(15, 10))
        grid = torchvision.utils.make_grid(nrow=20, tensor=torch.Tensor(imgs))
        # print(f"image tensor: {imgs.shape}")
        print(f"class labels: {lbls}")
        plt.imshow(np.transpose(grid, axes=(1, 2, 0)), cmap='gray')
        plt.show()

    return train_dataset, test_dataset, hess_dataset

def get_fashion_mnist_corrupt(flatten=False, tr_indices=60000, te_indices=10000, hess_indices=None,
                      tr_classes=10, te_classes=10, hess_classes=None, corrupt_p=0., one_hot=False,
                      visualise=True, rand_start=False):
    return get_fashion_mnist(flatten, tr_indices, te_indices, hess_indices, tr_classes, te_classes, hess_classes, corrupt_p, one_hot, visualise, rand_start)


def get_split_mnist(mode='f', flatten=False, tr_indices=60000, te_indices=10000, hess_indices=None,
                      classes=10, class_list=None, one_hot=False, visualise=True, rand_start=False):
    base_dataset = datasets.FashionMNIST if mode == 'f' else datasets.MNIST
# fashion only, regular mnist todo
    if class_list is not None:
        assert len(class_list) == classes
    else:
        class_list = [x for x in range(classes)]
    assert tr_indices // classes <= 6000
    assert te_indices // classes <= 1000

    normalize = transforms.Normalize((0.1307,), (0.3081,))  # MNIST
    if flatten:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            FlattenAndCast(),
        ])

    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
            CastNP_MNIST(),
        ])

    train_dataset = base_dataset(root=data_dir, train=True,
                download=True, transform=transform)
    hess_dataset = base_dataset(root=data_dir, train=True,
                download=True, transform=transform)
    test_dataset = base_dataset(root=data_dir, train=False,
                download=True, transform=transform)

    if hess_indices is not None:
        assert classes is not None
        # hv_idx = np.where(np.array(train_dataset.targets) in class_list)[0]
        hv_idx = np.where(np.isin(train_dataset.targets, class_list) > 0)[0]
        if hess_indices < 50000:
            if not rand_start:
                hv_idx = hv_idx[torch.arange(hess_indices)]
            else:
                hv_idx = hv_idx[torch.arange(hess_indices) + torch.randint(high=len(hess_dataset) - hess_indices - 1, size=(1,))]
        hess_inputs = np.array([hess_dataset.__getitem__(i)[0] for i in hv_idx])
        hess_targets = np.array([hess_dataset.__getitem__(i)[1] for i in hv_idx])
        if not one_hot:
            hess_dataset = Dataset(hess_inputs, hess_targets)
        else:
            hess_targets_oh = np.array(torch.nn.functional.one_hot(torch.tensor(hess_targets), num_classes=classes))  # change the original label y ?
            hess_dataset = Dataset(hess_inputs, hess_targets_oh)
    else:
        pass

    # print(np.array(train_dataset.targets).shape)
    # print(np.isin(train_dataset.targets, class_list))
    tr_idx = np.where(np.isin(train_dataset.targets, class_list) > 0)[0]
    te_idx = np.where(np.isin(test_dataset.targets, class_list) > 0)[0]

    # print(idx)
    if one_hot:
        train_dataset = data_utils.Subset(onehot_DS(train_dataset, classes), tr_idx)
        test_dataset = data_utils.Subset(onehot_DS(test_dataset, classes), te_idx)

    else:
        train_dataset = data_utils.Subset(train_dataset, tr_idx)
        test_dataset = data_utils.Subset(test_dataset, te_idx)

    # print(len(idx[0]))

    if tr_indices < 60000:
        if not rand_start:
            indices = torch.arange(tr_indices)
        else:
            indices = torch.arange(tr_indices) + torch.randint(high=len(train_dataset)-tr_indices-1, size=(1,))
        train_inputs = np.array([train_dataset.__getitem__(i)[0] for i in indices])
        train_targets = np.array([train_dataset.__getitem__(i)[1]-class_list[0] for i in indices])
        train_dataset = Dataset(train_inputs, train_targets)

    if te_indices < 10000:
        if not rand_start:
            indices = torch.arange(te_indices)
        else:
            indices = torch.arange(te_indices) + torch.randint(high=len(test_dataset)-te_indices-1, size=(1,))
        test_inputs = np.array([test_dataset.__getitem__(i)[0] for i in indices])
        test_targets = np.array([test_dataset.__getitem__(i)[1]-class_list[0] for i in indices])
        test_dataset = Dataset(test_inputs, test_targets)

    if visualise:
        raw_dataset = base_dataset(root=data_dir, train=True,
                download=True, transform=transforms.ToTensor())
        imgs = []
        lbls = []
        for i in range(10):
            idx = tr_idx[i]
            imgs.append(np.array(raw_dataset.data[idx])/255)
            lbls.append(raw_dataset.targets[idx])
            # if one_hot:
            #     lbls.append(np.argmax(corrupt_set[idx][1]))
            # else:
            #     lbls.append(corrupt_set[idx][1])
        # imgs, lbls = sample
        imgs = np.array(imgs)
        lbls = np.array(lbls)
        imgs = imgs.reshape(imgs.shape[0], 1, 28, 28)
        plt.figure(figsize=(15, 10))
        grid = torchvision.utils.make_grid(nrow=20, tensor=torch.Tensor(imgs))
        # print(f"image tensor: {imgs.shape}")
        print(f"class labels: {lbls}")
        plt.imshow(np.transpose(grid, axes=(1, 2, 0)), cmap='gray')
        plt.show()

    return train_dataset, test_dataset, hess_dataset


def get_fashion_mnist_corrupt(flatten=False, tr_indices=60000, te_indices=10000, hess_indices=None,
                      tr_classes=10, te_classes=10, hess_classes=None, corrupt_p=0., one_hot=False,
                      visualise=True, rand_start=False):
    return get_fashion_mnist(flatten, tr_indices, te_indices, hess_indices, tr_classes, te_classes, hess_classes, corrupt_p, one_hot, visualise, rand_start)


def get_tiny_shakespeare(block_size=32, tr_indices=90000, te_indices=10000, hess_indices=None, pred_gap=1):
    with open('data/tiny_shakespeare/input.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    chars = sorted(list(set(text)))
    vocab_size = len(chars)  # This is the number of possible elements of our sequence
    stoi = {ch: i for i, ch in enumerate(chars)}  # Dictionary that maps characters to integers
    itos = {i: ch for i, ch in enumerate(chars)}  # Dictionary that maps integers to characters
    encode = lambda s: [stoi[c] for c in s]  # encoder: take a string, output a list of integers
    decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string
    data = np.array(encode(text))  # Represent data into a large sequence of integers
    assert tr_indices + block_size + pred_gap + te_indices + block_size + pred_gap < len(data)

    train_x, train_y = [data[i:i+block_size] for i in range(tr_indices)], [data[i+pred_gap:i+block_size+pred_gap] for i in range(tr_indices)]
    test_x, test_y = [data[i:i + block_size] for i in range(te_indices)], [data[i + pred_gap:i + block_size + pred_gap] for i in range(te_indices)]
    train_ds = Dataset(np.array(train_x, dtype=np.float32)[:, :, np.newaxis], np.array(train_y))
    val_ds = Dataset(np.array(test_x, dtype=np.float32)[:, :, np.newaxis], np.array(test_y))

    return train_ds, val_ds