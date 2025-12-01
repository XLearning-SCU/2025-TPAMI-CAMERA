import os, random, sys
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
from sklearn.model_selection import train_test_split

# from datasets_MNISTUSPS import mnist_usps
# def normalize(x):
#     x = (x-np.mean(x)) / np.std(x)
#     return x

def load_data(args, **kwargs):
    """Load data """

    data_name = args.dataset_name
    main_dir = sys.path[0]
    label = []
    data=[]
    if data_name in ['Scene_15']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'Scene-15.mat'))
        # data = mat['X'][0][0:2]  # 20, 59 dimensions
        data.append(mat['X'][0][1])
        data.append(mat['X'][0][0])  # 20, 59 dimensions
        label = np.squeeze(mat['Y']) - 1

    elif data_name in ['LandUse_21']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', 'LandUse-21.mat'))
        train_x = []
        train_x.append(sparse.csr_matrix(mat['X'][0, 0]).A)  # 20
        train_x.append(sparse.csr_matrix(mat['X'][0, 1]).A)  # 59
        train_x.append(sparse.csr_matrix(mat['X'][0, 2]).A)  # 40
        index = random.sample(range(train_x[0].shape[0]), 2100)
        for view in [1, 2]:
            x = train_x[view][index]
            data.append(x)
        label=np.squeeze(mat['Y']).astype('int')[index]

    elif data_name in ['NoisyMNIST']:
        mat = sio.loadmat('./data/NoisyMNIST30000.mat')
        data.append(mat['X1'])
        data.append(mat['X2'])
        label = np.squeeze(mat['Y']) - 1

    elif data_name in ['Caltech101-20']:
        mat = sio.loadmat(os.path.join(main_dir, 'data', data_name + '.mat'))
        data = mat['X'][0][3:5]
        label = np.squeeze(mat['Y']) - 1

    elif data_name in ['Reuters']:
        mat = sio.loadmat('./data/' + "Reuters" + '.mat')
        data.append(np.vstack((mat['x_train'][0], mat['x_test'][0])))
        data.append(np.vstack((mat['x_train'][1], mat['x_test'][1])))
        label=(np.squeeze(np.hstack((mat['y_train'], mat['y_test']))))

    elif data_name in ['Deep-Caltech101']:
        mat = sio.loadmat('./data/' + "2view-caltech101-8677sample" + '.mat')
        data.append(mat['X'][0][0].T)
        data.append(mat['X'][0][1].T)
        label=(np.squeeze(mat['gt']) - 1)

    elif data_name in ['MNIST-USPS']:
        data0,data1,label_0=mnist_usps(args=args)
        data.append(data1.cpu().numpy())
        data.append(data0.cpu().numpy())
        label=np.asarray(label_0)

    elif data_name in ['Caltech101-all']:
        mat = sio.loadmat('./data/' + "Caltech101-all" + '.mat')
        data1=mat['X'][0][3]
        data2=mat['X'][0][4]
        data.append(data1)
        data.append(data2)
        label = np.squeeze(mat['Y'])

        
    elif data_name in ['Deep-Animal']:
        mat = sio.loadmat('./data/' + "Deep-Animal" + '.mat')
        data.append(mat['X'][0][6].T)
        data.append(mat['X'][0][5].T)
        label=(np.squeeze(mat['gt']))

    elif data_name in ['cub_googlenet']:
        mat = sio.loadmat('./data/' + "cub_googlenet_doc2vec_c10" + '.mat')
        data.append(mat['X'][0][0])
        data.append(mat['X'][0][1])
        label=(np.squeeze(mat['gt']))

    elif data_name in ['DHA', 'UWA30']:
        train_data = data_loader_HAR(data_name)
        train_data.read_train()
        train_data_x, train_data_y, test_data_x, test_data_y, label1= train_data.get_data()
        # print(train_data_y.shape,test_data_y.shape)
        # exit(-1)
        data.append(np.concatenate([train_data_y, test_data_y], axis=0))
        data.append(np.concatenate([train_data_x, test_data_x], axis=0))
        label=(np.argmax(label1,axis=1))


    # if 'MNIST' in data_name or 'cub' in data_name or 'Reuters' in data_name or 'Caltech101' in data_name:
    if 'MNIST' in data_name or 'cub' in data_name or 'Reuters' in data_name or 'Caltech101' in data_name or 'DHA' in data_name:
        print("sample_norm")
        data=data_normalize(data,normalize='sample')
    elif 'UWA30' in data_name:
        data=data
    else:
        data=data_normalize(data)

    train_data_list=[]
    test_data_list=[]
    all_data_np=np.concatenate((data[0],data[1]),axis=1)
    len_x=data[0].shape[1]

    if data_name in ['DHA', 'UWA30']:
        # x_train, x_test, train_label, test_label = train_test_split(all_data_np, label, test_size=0.5)
        train_data_list.append(train_data.train_data_y)
        train_data_list.append(train_data.train_data_x)
        test_data_list.append(train_data.test_data_y)
        test_data_list.append(train_data.test_data_x)
        train_label=np.argmax(train_data.train_data_label,axis=1)
        test_label=np.argmax(train_data.test_data_label,axis=1)
    else:
        x_train, x_test, train_label, test_label = train_test_split(all_data_np, label, test_size=0.2)
        train_data_list.append(x_train[:,:len_x])
        train_data_list.append(x_train[:,len_x:])
        test_data_list.append(x_test[:,:len_x])
        test_data_list.append(x_test[:,len_x:])

    train_data,train_label_x,train_label_y,train_mask,train_pair=get_sp_vp_data(train_data_list,train_label,args)
    test_data,test_label_x,test_label_y,test_mask,test_pair=get_sp_vp_data(test_data_list,test_label,args,test=False)

    class_num = len(np.unique(label))
    return train_data, train_label_x, train_label_y, train_mask, train_pair, test_data, test_label_x, test_label_y, test_mask, test_pair, class_num


def get_sp_vp_data(data,label,args,test=False):
    if test==True:
        args.complete_prop=1.0
        args.aligned_prop=1.0
    all_data=[]
    divide_seed = np.random.randint(1, 1000)
    train_idx, test_idx = TT_split(len(label), 1 - args.aligned_prop, divide_seed)
    train_label, test_label = label[train_idx], label[test_idx]

    train_X, train_Y, test_X, test_Y = data[0][train_idx], data[1][train_idx], data[0][test_idx], data[1][test_idx]
    pair_sample = len(train_X)

    # Use test_prop*sizeof(all data) to train the MvCLN, and shuffle the rest data to simulate the unargs.aligned data.
    # Note that, MvCLN establishes the correspondence of the all data rather than the unargs.aligned portion in the testing.
    # When test_prop = 0, MvCLN is directly performed on the all data without shuffling.
    if args.aligned_prop == 1:
        all_data.append(train_X)
        all_data.append(train_Y)
        all_label_X, all_label_Y = train_label, train_label
    else:
        shuffle_idx = random.sample(range(len(test_Y)), len(test_Y))
        test_Y = test_Y[shuffle_idx]
        test_label_X, test_label_Y = test_label, test_label[shuffle_idx]
        all_data.append(np.concatenate((train_X, test_X)))
        all_data.append(np.concatenate((train_Y, test_Y)))
        all_label_X = np.concatenate((train_label, test_label_X))
        all_label_Y = np.concatenate((train_label, test_label_Y))

    test_mask = get_sn(2, len(test_label), 1 - args.complete_prop)
    if args.aligned_prop == 1.:
        mask = test_mask
    else:
        identy_mask = np.ones((len(train_label), 2))
        mask = np.concatenate((identy_mask, test_mask))

    return all_data, all_label_X, all_label_Y, mask, pair_sample

    # return data, label
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

    labels = np.array(labels, dtype=np.int64)
    real_labels = np.array(real_labels, dtype=np.int64)
    class_labels0, class_labels1 = np.array(class_labels0, dtype=np.int64), np.array(class_labels1, dtype=np.int64)
    view0 = np.array(view0, dtype=np.float32)
    view1 = np.array(view1, dtype=np.float32)
    return view0, view1, labels, real_labels, class_labels0, class_labels1

def get_sn(view_num, alldata_len, missing_rate):
    """Randomly generate incomplete data information, simulate partial view data with complete view data
    :param view_num:view number
    :param alldata_len:number of samples
    :param missing_rate:Defined in section 4.3 of the paper
    :return:Sn
    """
    missing_rate = missing_rate / 2
    one_rate = 1.0 - missing_rate
    if one_rate <= (1 / view_num):
        enc = OneHotEncoder()  # n_values=view_num
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        return view_preserve
    error = 1
    if one_rate == 1:
        matrix = randint(1, 2, size=(alldata_len, view_num))
        return matrix
    while error >= 0.005:
        enc = OneHotEncoder()  # n_values=view_num
        view_preserve = enc.fit_transform(randint(0, view_num, size=(alldata_len, 1))).toarray()
        one_num = view_num * alldata_len * one_rate - alldata_len
        ratio = one_num / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
        a = np.sum(((matrix_iter + view_preserve) > 1).astype(np.int))
        one_num_iter = one_num / (1 - a / one_num)
        ratio = one_num_iter / (view_num * alldata_len)
        matrix_iter = (randint(0, 100, size=(alldata_len, view_num)) < int(ratio * 100)).astype(np.int)
        matrix = ((matrix_iter + view_preserve) > 0).astype(np.int)
        ratio = np.sum(matrix) / (view_num * alldata_len)
        error = abs(one_rate - ratio)
    return matrix

class getAllDataset(Dataset):
    def __init__(self, data, class_labels0, class_labels1, mask, pair_sample, transforms=None):
        self.data = data
        self.class_labels0 = class_labels0
        self.class_labels1 = class_labels1
        self.mask = mask == 1
        self.labels=class_labels0
        self.pair_sample = pair_sample
        if transforms is None:
            transforms = [None, None]
        self.transforms = transforms

    def __getitem__(self, index):
        fea0, fea1 = (torch.from_numpy(self.data[0][index])).type(torch.FloatTensor), (
            torch.from_numpy(self.data[1][index])).type(torch.FloatTensor)
        class_labels0 = np.int64(self.class_labels0[index])
        class_labels1 = np.int64(self.class_labels1[index])
        mask = self.mask[index]
        align_pair = self.pair_sample > index
        return fea0, fea1, class_labels0, class_labels1, mask, align_pair, index

    def __len__(self):
        return len(self.labels)

def get_dataset(args, **kwargs):
    train_data,train_label_x,train_label_y,train_mask,train_pair,test_data,test_label_x,test_label_y,test_mask,test_pair,class_num = load_data(args)
    train_dataset = getAllDataset(train_data,train_label_x,train_label_y,train_mask,train_pair)
    # test_dataset = getAllDataset(train_data,train_label_x,train_label_y,train_mask,test_pair)
    test_dataset = getAllDataset(test_data,test_label_x,test_label_y,test_mask,test_pair)
    return train_dataset,test_dataset, class_num

def get_loader(data_set, batch_size,drop_last=True):
    train_loader = data.DataLoader(data_set,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=drop_last)
    eval_loader = data.DataLoader(data_set, batch_size=1024,shuffle=False,drop_last=False)
    index_loader = data.DataLoader(data_set, batch_size=1000*batch_size)
    return train_loader, eval_loader,index_loader

def get_dataloader(args, **kwargs):
    # print('getting data {}'.format(args.dataset))
    # if args.dataset_name == 'MNIST-USPS':
    #     data_set, class_num = mnist_usps(args=args)
    # else:
    train_dataset,test_dataset, class_num = get_dataset(args=args, **kwargs)
    if args.dataset_name =="cub_googlenet" or args.dataset_name =="DHA" or args.dataset_name =="UWA30":
        drop_last=False
    else:
        drop_last=True
    train_loader, eval_train_loader, index_loader = get_loader(train_dataset, args.batch_size,drop_last)
    eval_test_loader=data.DataLoader(test_dataset, batch_size=1024,shuffle=False,drop_last=False)
    # print('got data {}'.format(args.dataset))
    return train_loader, eval_train_loader, index_loader, eval_test_loader, class_num

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
            # x=(x-mean)/(std)
        x = x.view((-1, x.shape[-1])).numpy()
        trans_x.append(x)
    return trans_x

def mnist_usps(args):
    # print(path_operator.get_dataset_path('MNIST', level=2),)
    main_dir = sys.path[0]
    root_MNIST=os.path.join(main_dir, 'data', 'MNIST')
    root_USPS=os.path.join(main_dir, 'data', 'USPS')
    if args.dataset_name == 'MNIST-USPS':
        d0 = torchvision.datasets.MNIST(
            train=1,
            download=False,
            root=root_MNIST,
            transform=None)
        d1 = torchvision.datasets.USPS(
            train=True,
            download=False,
            root=root_USPS,
            transform=None)
        DataSource=False
        Image01AndTanh=False
        Image01NoTanh=True
        MaxSampleNum=5000
        if DataSource:
            data = np.load(args.DataSource)
            d0.data = torch.from_numpy(data['d0_data'])
            d0.targets = data['d0_targets']
            d1.data = torch.from_numpy(data['d1_data'])
            d1.targets = data['d1_targets']
        else:
            def trans(x):
                res = []
                for x0 in x:
                    x0 = Image.fromarray(x0)
                    x0 = torchvision.transforms.Resize(28)(x0)
                    x0 = torchvision.transforms.ToTensor()(x0)
                    res.append(x0)
                x = torch.cat(res)
                if Image01NoTanh:
                    mi, ma = torch.amin(x), torch.amax(x)
                    x = torchvision.transforms.Normalize([(mi + ma) / 2], [(ma - mi) / 2])(x)
                else:
                    mean, std = torch.mean(x), torch.std(x)
                    x = torchvision.transforms.Normalize([mean], [std])(x)
                    x = torch.nn.Tanh()(x)
                x = torch.unsqueeze(x, dim=1)
                return x

            x1 = trans(d0.data.numpy())
            x2 = trans(d1.data)
            
            if Image01AndTanh:
                x_ = torch.cat([x1, x2])
                mean, std = torch.mean(x_), torch.std(x_)
                x_ = torchvision.transforms.Normalize([mean], [std])(x_)
                x_ = torch.nn.Tanh()(x_)
                x1 = x_[:len(x1)]
                x2 = x_[-len(x2):]
            d0.data = x1
            d1.data = x2

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
        label_0=d0.targets
        label_1=d1.targets
        return d0.data,d1.data,label_0
    #     class_num = 10
    #     ds = GetDataset([d0, d1], args)
    # else:
    #     raise NotImplementedError('')
    # return ds, class_num

# class GetDataset(Dataset):
#     def __init__(self, ds, args):
#         self.datasets = ds
#         super(GetDataset, self).__init__()
#         self.is_pair = torch.zeros(len(ds[0].data), dtype=torch.int)
#         # align
#         aligned_len = int(len(self.is_pair) * args.aligned_prop)
#         aligned = np.arange(len(self.is_pair))
#         np.random.shuffle(aligned)
#         self.is_pair[aligned[:aligned_len]] = 1
#         self.is_pair = self.is_pair == 1
#         test_mask = get_sn(2, len(self.is_pair) - aligned_len, 1 - args.complete_prop)
#         if aligned_len == 0:
#             mask = test_mask
#         else:
#             mask = np.ones((len(self.is_pair), 2), dtype=int)
#             mask[aligned[aligned_len:]] = test_mask
#         self.mask = np.asarray(mask, dtype=int)
#         self.mask = self.mask == 1
#     def __getitem__(self, index):
#         xs = []
#         ys = []
#         for dset in self.datasets:
#             x = dset.data[index % len(dset)]
#             y = int(dset.targets[index % len(dset)])
#             xs.append(x)
#             ys.append(y)
#         mask = self.mask[index]
#         is_pair = self.is_pair[index]
#         return xs, ys, mask, is_pair, index

#     def __len__(self):
#         return np.max([len(dset) for dset in self.datasets])

class data_loader_HAR:

    def __init__(self, database_name='DHA'):

        self.filename = database_name
        self.data_x = []  # training and testing RGB + depth feature
        self.data_label = []  # training and testing depth label
        self.train_data_x = []  # training depth feature
        self.train_data_y = []  # training RGB feature
        self.train_data_label = []  # training label
        self.test_data_x = []  # testing depth feature
        self.test_data_y = []  # testing RGB feature
        self.test_data_label = []  # testing label
        # self.train_data_xy = []  # training RGB + depth feature
        # self.test_data_xy = []  # testing RGB + depth feature
        self.cluster = 0

    def read_train(self):

        # Depth feature -> 110-dimension
        # RGB feature -> 3x2048 dimension
        feature_num1 = 110
        feature_num2 = 3 * 2048
        feature_num = feature_num1 + feature_num2
        num = 0

        # load .csv file for training
        f = open('data/' + self.filename + '_total_train.csv', 'r')
        for i in f:
            num += 1
            row1 = i.rstrip().split(',')[:-1]
            row = [float(x) for x in row1]
            self.data_x.append(row[0:feature_num])
            self.data_label.append(row[feature_num:])
            self.train_data_x.append(row[0:feature_num1])
            self.train_data_y.append(row[feature_num1:feature_num1 + feature_num2])
            # self.train_data_xy.append(row[0:feature_num1 + feature_num2])
            self.train_data_label.append(row[feature_num1 + feature_num2:])
        f.close()

        # load .csv file for training
        f = open('data/' + self.filename + '_total_test.csv', 'r')
        for i in f:
            num += 1
            row1 = i.rstrip().split(',')[:-1]
            row = [float(x) for x in row1]
            self.data_x.append(row[0:feature_num])
            self.data_label.append(row[feature_num:])
            self.test_data_x.append(row[0:feature_num1])
            self.test_data_y.append(row[feature_num1:feature_num1 + feature_num2])
            # self.test_data_xy.append(row[0:feature_num1 + feature_num2])
            self.test_data_label.append(row[feature_num1 + feature_num2:])
        f.close()

        # random split training and test data
        train_data_x, train_data_y, test_data_x, test_data_y, label = self.get_data()
        data_x = np.concatenate([train_data_x, test_data_x], axis=0)
        data_y = np.concatenate([train_data_y, test_data_y], axis=0)
        self.train_data_x, self.test_data_x, self.train_data_y, self.test_data_y, self.train_data_label, self.test_data_label = train_test_split(
            data_x, data_y, label, test_size=0.5)

        # got the sample number
        self.sample_total_num = len(self.data_x)
        self.sample_train_num = len(self.train_data_x)
        self.sample_test_num = len(self.test_data_x)

        self.cluster = len(self.data_label[0])

    def get_data(self):
        train_data_x = np.array(self.train_data_x)
        train_data_y = np.array(self.train_data_y)
        test_data_x = np.array(self.test_data_x)
        test_data_y = np.array(self.test_data_y)

        label = np.concatenate([self.train_data_label, self.test_data_label], axis=0)
        # # label_new = [np.argmax(one_hot) for one_hot in label]
        # train_data_label=self.train_data_label
        # test_data_label=self.test_data_label

        return train_data_x, train_data_y, test_data_x, test_data_y, label,

    # randomly choose _batch_size RGB and depth feature in the training set
    def train_next_batch(self, _batch_size):
        xx = []  # training batch of depth features
        yy = []  # training batch of RGB features
        zz = []  # training batch of labels
        for sample_num in random.sample(range(self.sample_train_num), _batch_size):
            xx.append(self.train_data_x[sample_num])
            yy.append(self.train_data_y[sample_num])
            zz.append(self.train_data_label[sample_num])
        return yy, xx, zz

    # randomly choose _batch_size RGB and depth feature in the testing set
    def test_next_batch(self, _batch_size):
        xx = []  # testing batch of depth features
        yy = []  # testing batch of RGB features
        zz = []  # testing batch of labels
        for sample_num in random.sample(range(self.sample_test_num), _batch_size):
            xx.append(self.test_data_x[sample_num])
            yy.append(self.test_data_y[sample_num])
            zz.append(self.test_data_label[sample_num])
        return yy, xx, zz