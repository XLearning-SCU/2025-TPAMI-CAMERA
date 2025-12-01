import os
import random
import sys
import numpy as np
import scipy.io as sio
import util
from scipy import sparse
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from util import TT_split
from numpy.random import randint
from torch.utils import data
from sklearn.preprocessing import OneHotEncoder
from PIL import Image


def load_data(args, **kwargs):
    """Load data."""
    data_name = args.dataset_name
    main_dir = sys.path[0]
    data = []
    label = []

    if data_name == 'Scene_15':
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'Scene-15.mat'))
        data.append(mat['X'][0][1])
        data.append(mat['X'][0][0])
        label = np.squeeze(mat['Y']) - 1

    elif data_name == 'LandUse_21':
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'LandUse-21.mat'))
        train_x = []
        train_x.append(sparse.csr_matrix(mat['X'][0, 0]).A)
        train_x.append(sparse.csr_matrix(mat['X'][0, 1]).A)
        train_x.append(sparse.csr_matrix(mat['X'][0, 2]).A)
        index = random.sample(range(train_x[0].shape[0]), 2100)
        for view in [1, 2]:
            x = train_x[view][index]
            data.append(x)
        label = np.squeeze(mat['Y']).astype('int')[index]

    elif data_name == 'NoisyMNIST':
        mat = sio.loadmat('./data/NoisyMNIST30000.mat')
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y']) - 1

    elif data_name == 'Reuters':
        mat = sio.loadmat('./data/' + "Reuters" + '.mat')
        data.append(np.vstack((mat['x_train'][0], mat['x_test'][0])))
        data.append(np.vstack((mat['x_train'][1], mat['x_test'][1])))
        label = np.squeeze(np.hstack((mat['y_train'], mat['y_test'])))

    elif data_name == 'MNIST-USPS':
        mat = sio.loadmat('./data/' + "MNIST-USPS" + '.mat')
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y'])

    elif data_name == 'cub_googlenet':
        mat = sio.loadmat('./data/' + "cub_googlenet_doc2vec_c10" + '.mat')
        data.append(mat['X'][0][0])
        data.append(mat['X'][0][1])
        label = np.squeeze(mat['gt'])

    if 'MNIST' in data_name or 'cub' in data_name or 'Reuters' in data_name or 'Caltech101' in data_name:
        print("sample_norm")
        data = data_normalize(data, normalize='sample')
    else:
        data = data_normalize(data)

    divide_seed = np.random.randint(1, 1000)
    train_idx, test_idx = TT_split(len(label), 1 - args.aligned_prop, divide_seed)
    train_label, test_label = label[train_idx], label[test_idx]
    train_X, train_Y, test_X, test_Y = data[0][train_idx], data[1][train_idx], data[0][test_idx], data[1][test_idx]
    pair_sample = len(train_X)

    # Follow Robust Multi-view Clustering with Incomplete Information, TPAMI, 2022
    if args.aligned_prop == 1:
        all_data = [train_X, train_Y]
        all_label, all_label_X, all_label_Y = train_label, train_label, train_label
    else:
        test_label_X, test_label_Y = test_label, test_label
        all_data = [np.concatenate((train_X, test_X)), np.concatenate((train_Y, test_Y))]
        all_label = np.concatenate((train_label, test_label))
        all_label_X = np.concatenate((train_label, test_label_X))
        all_label_Y = np.concatenate((train_label, test_label_Y))

    test_mask = get_sn(2, len(test_label), 1 - args.complete_prop)
    if args.aligned_prop == 1.:
        mask = test_mask
    else:
        identy_mask = np.ones((len(train_label), 2))
        mask = np.concatenate((identy_mask, test_mask))

    return all_data, all_label, all_label_X, all_label_Y, mask, pair_sample


def get_pairs(train_X, train_Y, train_label):
    view0, view1, labels, real_labels, class_labels0, class_labels1 = [], [], [], [], [], []
    # construct pos. pairs
    for i in range(len(train_X)):
        view0.append(train_X[i])
        view1.append(train_Y[i])
        labels.append(1)
        real_labels.append(1)
        class_labels0.append(train_label[i])
        class_labels1.append(train_label[i])

    labels = np.array(labels, dtype=int)
    real_labels = np.array(real_labels, dtype=int)
    class_labels0 = np.array(class_labels0, dtype=int)
    class_labels1 = np.array(class_labels1, dtype=int)
    view0 = np.array(view0, dtype=np.float32)
    view1 = np.array(view1, dtype=np.float32)
    return view0, view1, labels, real_labels, class_labels0, class_labels1


def get_sn(view_num, alldata_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    :param view_num: view number
    :param alldata_len: number of samples
    :param missing_rate: Defined in section 4.3 of the paper
    :return: Sn
    """
    missing_rate = missing_rate / 2
    one_rate = 1.0 - missing_rate
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1:
        matrix = randint(1, 2, size=(alldata_len, view_num))
        return matrix
    while error >= 0.005:
        enc = OneHotEncoder()
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        one_num = view_num * alldata_len * one_rate - alldata_len
        ratio = one_num / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(int))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)
    return matrix


class getAllDataset(Dataset):
    def __init__(self, data, labels, class_labels0, class_labels1, mask, pair_sample, transforms=None):
        self.data = data
        self.labels = labels
        self.class_labels0 = class_labels0
        self.class_labels1 = class_labels1
        self.mask = mask == 1
        self.pair_sample = pair_sample
        if transforms is None:
            transforms = [None, None]
        self.transforms = transforms

    def __getitem__(self, index):
        fea0 = torch.from_numpy(self.data[0][index]).type(torch.FloatTensor)
        fea1 = torch.from_numpy(self.data[1][index]).type(torch.FloatTensor)
        label = int(self.labels[index])
        class_labels0 = int(self.class_labels0[index])
        class_labels1 = int(self.class_labels1[index])
        mask = self.mask[index]
        align_pair = self.pair_sample > index
        return fea0, fea1, class_labels0, class_labels1, mask, align_pair, index

    def __len__(self):
        return len(self.labels)


def get_dataset(args, **kwargs):
    all_data, all_label, all_label_X, all_label_Y, mask, pair_sample = load_data(args)
    all_dataset = getAllDataset(all_data, all_label, all_label_X, all_label_Y, mask, pair_sample)
    class_num = len(np.unique(all_label_X))
    return all_dataset, class_num


def get_loader(data_set, batch_size, drop_last=True):
    train_loader = data.DataLoader(data_set, batch_size=batch_size, shuffle=True, drop_last=drop_last)
    test_loader = data.DataLoader(data_set, batch_size=1024, shuffle=False, drop_last=False)
    index_loader = data.DataLoader(data_set, batch_size=1000 * batch_size)
    return train_loader, test_loader, index_loader


def get_dataloader(args, **kwargs):
    data_set, class_num = get_dataset(args=args, **kwargs)
    drop_last = False if args.dataset_name == "cub_googlenet" else True
    train_loader, test_loader, index_loader = get_loader(data_set, args.batch_size, drop_last)
    return train_loader, test_loader, index_loader, class_num


def data_normalize(data, normalize='dim'):
    trans_x = []
    for x in data:
        x = torch.from_numpy(x)
        x = x.view((-1, 1, 1, x.shape[-1]))
        if normalize == 'dim':
            mean, std = torch.mean(x, dim=0), torch.std(x, dim=0)
            std[std < torch.max(std) * 1e-6] = 1
            x = torchvision.transforms.Normalize(mean, std)(x)
        elif normalize == 'sample':
            mean, std = torch.mean(x), torch.std(x)
            x = torchvision.transforms.Normalize(mean, std)(x)
        x = x.view((-1, x.shape[-1])).numpy()
        trans_x.append(x)
    return trans_x


def mnist_usps(args):
    main_dir = sys.path[0]
    root_MNIST = os.path.join(main_dir, 'data', 'MNIST')
    root_USPS = os.path.join(main_dir, 'data', 'USPS')
    d0 = torchvision.datasets.MNIST(train=True, download=False, root=root_MNIST, transform=None)
    d1 = torchvision.datasets.USPS(train=True, download=False, root=root_USPS, transform=None)

    def trans(x):
        res = []
        for x0 in x:
            x0 = Image.fromarray(x0)
            x0 = torchvision.transforms.Resize(28)(x0)
            x0 = torchvision.transforms.ToTensor()(x0)
            res.append(x0)
        x = torch.cat(res)
        mean, std = torch.mean(x), torch.std(x)
        x = torchvision.transforms.Normalize([mean], [std])(x)
        x = torch.unsqueeze(x, dim=1)
        return x

    x1 = trans(d0.data.numpy())
    x2 = trans(d1.data)

    MaxSampleNum = 5000

    def re_sample(dx):
        classes = np.unique(dx.targets)
        num_per_class = int(MaxSampleNum / len(classes))
        ind = []
        for y in classes:
            nd = np.arange(len(dx.targets))[dx.targets == y]
            np.random.shuffle(nd)
            ind.extend(list(nd[:num_per_class]))
        dx.data = dx.data[ind]
        dx.targets = np.asarray(dx.targets)[ind]

    if len(d0.data) > MaxSampleNum:
        re_sample(d0)
    if len(d1.data) > MaxSampleNum:
        re_sample(d1)

    d0.data = d0.data.view((len(d0.data), -1))
    d1.data = d1.data.view((len(d1.data), -1))
    label_0 = d0.targets
    return d0.data, d1.data, label_0