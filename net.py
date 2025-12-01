import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, in_dims, class_num, c_num=2):

        super(Net, self).__init__()
        if in_dims[0]!=in_dims[1]:
            self.GroupWiseLayer='100111'
        else:
            self.GroupWiseLayer='000111'
        self.GroupWiseLayer='100111'
        self.BatchNormType='11001'
        # self.BatchNormType='11100'
        self.Dropout=0.2
        self.ElActivationType="Normalize"
        self.ActivationType='None'
        self.representation_dim=0
        self.McDecoder=False
        self.encoder_adaption = nn.ModuleList([
            get_ffn([in_dims[i], 1024], with_bn=self.BatchNormType[0] == '1', drop_out=self.Dropout)
            for i in range(c_num if self.GroupWiseLayer[0] == '1' else 1)])
        self.encoder_ = nn.ModuleList([
            get_ffn([1024,1024, 512], with_bn=self.BatchNormType[1] == '1', drop_out=self.Dropout)
            for _ in range(c_num if self.GroupWiseLayer[1] == '1' else 1)])
        if self.representation_dim == 0:
            self.representation_dim = class_num
        self.class_num = class_num
        self.c_num = c_num

        if self.ElActivationType == 'None':
            el_activation_ = []
        elif self.ElActivationType == 'Normalize':
            el_activation_ = []
        elif self.ElActivationType == 'BnNormalize':
            el_activation_ = [nn.BatchNorm1d(self.representation_dim)]
        elif self.ElActivationType == 'BnReNormalize':
            el_activation_ = [nn.BatchNorm1d(self.representation_dim), nn.ReLU()]
        elif self.ElActivationType == 'BnRe':
            el_activation_ = [nn.BatchNorm1d(self.representation_dim), nn.ReLU()]
        else:
            raise NotImplementedError('')
        self.el_activation_ = el_activation_
        self.encoder_linear = nn.ModuleList([
            get_ffn([512, 256], with_bn=self.BatchNormType[2] == '1', drop_out=self.Dropout,
                    last_layers=[nn.Linear(256, self.representation_dim)] + self.el_activation_)
            for _ in range(c_num if self.GroupWiseLayer[2] == '1' else 1)])
        dec_in = self.representation_dim
        if self.McDecoder:
            dec_in *= c_num
        self.dec_in = dec_in
        self.decoder_linear = nn.ModuleList([
            get_ffn([self.dec_in, 256, 512], with_bn=self.BatchNormType[3] == '1', drop_out=self.Dropout)
            for _ in range(c_num if self.GroupWiseLayer[3] == '1' else 1)])

        if self.ActivationType == 'None':
            final_activation_ = []
        elif self.ActivationType == 'Sigmoid':
            final_activation_ = [nn.Sigmoid()]
        elif self.ActivationType == 'Tanh':
            final_activation_ = [nn.Tanh()]
        else:
            raise NotImplementedError('')
        self.final_activation_ = final_activation_
        self.decoder_ = nn.ModuleList([
            get_ffn([512, 1024, 1024], with_bn=self.BatchNormType[4] == '1', drop_out=self.Dropout)
            for _ in range(c_num if self.GroupWiseLayer[4] == '1' else 1)])

        self.decoder_adaption = nn.ModuleList([
            get_ffn([], last_layers=[nn.Linear(1024, in_dims[i])] + self.final_activation_)
            for i in range(c_num if self.GroupWiseLayer[5] == '1' else 1)])

    def encoder(self,x1,x2):
        if self.GroupWiseLayer[0]=='1':
            latent1=self.encoder_adaption[0](x1)
            latent2=self.encoder_adaption[1](x2)
        else:
            latent1=self.encoder_adaption[0](x1)
            latent2=self.encoder_adaption[0](x2)
        if self.GroupWiseLayer[1]=='1':
            latent1=self.encoder_[0](latent1)
            latent2=self.encoder_[1](latent2)
        else:
            latent1=self.encoder_[0](latent1)
            latent2=self.encoder_[0](latent2)
        if self.GroupWiseLayer[2]=='1':
            latent1=self.encoder_linear[0](latent1)
            latent2=self.encoder_linear[1](latent2)
        else:
            latent1=self.encoder_linear[0](latent1)
            latent2=self.encoder_linear[0](latent2)
        
        return F.normalize(latent1,dim=1),F.normalize(latent2,dim=1)
    def decoder(self, latent1, latent2):
        latent1=self.decoder_linear[0](latent1)
        latent2=self.decoder_linear[1](latent2)
        latent1=self.decoder_[0](latent1)
        latent2=self.decoder_[1](latent2)
        latent1=self.decoder_adaption[0](latent1)
        latent2=self.decoder_adaption[1](latent2)
        return latent1,latent2

def get_ffn(dims, last_layers=None, with_bn=False, drop_out=0):
    layers = []
    for ind in range(len(dims) - 1):
        in_dim = dims[ind]
        out_dim = dims[ind + 1]
        layers.append(nn.Linear(in_dim, out_dim))
        if with_bn:
            layers.append(nn.BatchNorm1d(out_dim))
        layers.append(nn.ReLU())
        if drop_out:
            layers.append(nn.Dropout(drop_out))
    if last_layers is not None:
        layers.extend(last_layers)
    return nn.Sequential(*layers)

class VecToPic(nn.Module):
    def __init__(self, depth=1):
        super(VecToPic, self).__init__()
        self.depth = depth

    def __call__(self, x):
        dim = int(np.sqrt(x.shape[1] / self.depth))
        x = x.view((len(x), self.depth, dim, dim))
        return x

class NetCov(Net):
    def __init__(self, in_dims, class_num, c_num):
        super(NetCov, self).__init__(in_dims, class_num, c_num=2)
        if in_dims[0]!=in_dims[1]:
            self.GroupWiseLayer='000111'
        else:
            self.GroupWiseLayer='000111'
        self.GroupWiseLayer='000111'
        self.BatchNormType='00000'#ori00000
        self.cov_Dropout = 0.
        self.ffn_Dropout = 0. #2
        if self.representation_dim == 0:
            self.representation_dim = class_num

        self.encoder_adaption = nn.ModuleList([
            get_cov([], [], last_layers=[VecToPic()])
            for _ in range(c_num if self.GroupWiseLayer[0] == '1' else 1)])
        self.encoder_ = nn.ModuleList([
            get_cov([1, 16, 32, 32, 16], [1, 2, 1, 2], with_bn=self.BatchNormType[1] == '1', drop_out=self.cov_Dropout,
                    last_layers=[nn.Flatten()])
            for _ in range(c_num if self.GroupWiseLayer[1] == '1' else 1)])
        self.encoder_linear = nn.ModuleList([
            get_ffn([in_dims[i], 256], with_bn=self.BatchNormType[2] == '1', drop_out=self.ffn_Dropout,
                    last_layers=[nn.Linear(256, self.representation_dim)])
            for i in range(c_num if self.GroupWiseLayer[2] == '1' else 1)])
        self.decoder_linear = nn.ModuleList([
            get_ffn([self.dec_in, 256, in_dims[i]], with_bn=self.BatchNormType[3] == '1', drop_out=self.ffn_Dropout,
                    last_layers=[VecToPic(depth=16)])
            for i in range(c_num if self.GroupWiseLayer[3] == '1' else 1)])
        self.decoder_ = nn.ModuleList([
            get_cov([16, 32, 32, 16], [-2, -1, -2], with_bn=self.BatchNormType[4] == '1', drop_out=self.cov_Dropout)
            for _ in range(c_num if self.GroupWiseLayer[4] == '1' else 1)])
        self.decoder_adaption = nn.ModuleList([
            get_cov([], [], last_layers=[nn.ConvTranspose2d(16, 1, kernel_size=3, stride=1, padding=1),
                                         nn.Flatten()])
            for _ in range(c_num if self.GroupWiseLayer[5] == '1' else 1)])
    def encoder(self,x1,x2):
        latent1=self.encoder_adaption[0](x1)
        latent2=self.encoder_adaption[0](x2)
        latent1=self.encoder_[0](latent1)
        latent2=self.encoder_[0](latent2)
        latent1=self.encoder_linear[0](latent1)
        latent2=self.encoder_linear[0](latent2)
        return F.normalize(latent1,dim=1),F.normalize(latent2,dim=1)
    
    def decoder(self, latent1, latent2):
        if self.GroupWiseLayer[3]=='1':
            latent1=self.decoder_linear[0](latent1)
            latent2=self.decoder_linear[1](latent2)
        else:
            latent1=self.decoder_linear[0](latent1)
            latent2=self.decoder_linear[0](latent2)
        if self.GroupWiseLayer[4]=='1':
            latent1=self.decoder_[0](latent1)
            latent2=self.decoder_[1](latent2)
        else:
            latent1=self.decoder_[0](latent1)
            latent2=self.decoder_[0](latent2)
        if self.GroupWiseLayer[5]=='1':
            latent1=self.decoder_adaption[0](latent1)
            latent2=self.decoder_adaption[1](latent2)
        else:
            latent1=self.decoder_adaption[0](latent1)
            latent2=self.decoder_adaption[0](latent2)
        return latent1,latent2
def get_cov(dims, strides, last_layers=None, with_bn=False, drop_out=0):
    layers = []
    for ind in range(len(dims) - 1):
        in_dim = dims[ind]
        out_dim = dims[ind + 1]
        stride = strides[ind]
        # layers.append(nn.Linear(in_dim, out_dim))
        if stride >= 0:
            layers.append(nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=stride, padding=1))
        else:
            layers.append(nn.ConvTranspose2d(
                in_dim, out_dim, kernel_size=3, stride=-stride, padding=1, output_padding=0 if stride == -1 else 1))
        if with_bn:
            # layers.append(nn.BatchNorm1d(out_dim))
            layers.append(nn.BatchNorm2d(out_dim))
        layers.append(nn.ReLU())
        if drop_out:
            layers.append(nn.Dropout(drop_out))
    if last_layers is not None:
        layers.extend(last_layers)
    return nn.Sequential(*layers)