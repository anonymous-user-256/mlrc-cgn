"""Defines a generator network for MNIST."""

import numpy as np
import torch
from torch import nn

from utils import get_norm_layer, init_net, choose_rand_patches, Patch2Image, RandomCrop


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

def UpsampleBlock(cin, cout, scale_factor=2):
    return [
        nn.Upsample(scale_factor=scale_factor),
        nn.Conv2d(cin, cout, 3, stride=1, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.LeakyReLU(0.2, inplace=True),
    ]

def lin_block(in_feat, out_feat, normalize=True):
    layers = [nn.Linear(in_feat, out_feat)]
    layers.append(nn.BatchNorm1d(out_feat, 0.8))
    layers.append(nn.LeakyReLU(0.2, inplace=True))
    return layers

def shape_layers(cin, cout, ngf, init_sz):
    return [
        nn.Linear(cin, ngf*2 * init_sz ** 2),
        Reshape(*(-1, ngf*2, init_sz, init_sz)),
        get_norm_layer()(ngf*2),
        *UpsampleBlock(ngf*2, ngf),
        get_norm_layer()(ngf),
        *UpsampleBlock(ngf, cout),
        get_norm_layer()(cout),
    ]


class Generator(nn.Module):
    def __init__(self, n_classes=10, latent_sz=32, ngf=32,
                 init_type='orthogonal', init_gain=0.1, img_sz=32):
        super(Generator, self).__init__()

        # params
        self.batch_size = 1  # default: sample a single image
        self.n_classes = n_classes
        self.latent_sz = latent_sz
        self.label_emb = nn.Embedding(n_classes, n_classes)
        init_sz = img_sz // 4
        inp_dim = self.latent_sz + self.n_classes

        self.model = nn.Sequential(*shape_layers(inp_dim, 3, ngf, init_sz))

        init_net(self, init_type=init_type, init_gain=init_gain)

    def get_inp(self, ys):
        u_vec = torch.normal(0, 1, (len(ys), self.latent_sz)).to(ys.device)
        y_vec = self.label_emb(ys)
        return torch.cat([u_vec, y_vec], -1)

    def forward(self, ys=None, counterfactual=False):

        # create input
        inp = self.get_inp(ys)

        # generator
        x_gen = self.model(inp)

        return x_gen


class GenLin(nn.Module):
    def __init__(self, n_classes=10, latent_sz=32, ngf=32, img_shape=[3, 32, 32]):
        super(GenLin, self).__init__()

        self.n_classes = n_classes
        self.latent_sz = latent_sz
        inp_dim = self.latent_sz + self.n_classes
        self.img_shape = img_shape

        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.model = nn.Sequential(
            nn.Linear(inp_dim, 4 * ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(4 * ngf, 8 * ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(8 * ngf, 16 * ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(16 * ngf, int(np.prod(img_shape))),
            # nn.Tanh()
        )

    def get_inp(self, ys):
        u_vec = torch.normal(0, 1, (len(ys), self.latent_sz)).to(ys.device)
        y_vec = self.label_emb(ys)
        return torch.cat([u_vec, y_vec], -1)

    def forward(self, ys=None, counterfactual=False):
        inp = self.get_inp(ys)
        x_gen = self.model(inp)
        x_gen = x_gen.view(x_gen.size(0), *self.img_shape)
        return x_gen


class GenConv(nn.Module):
    """Convolutions generator.
    Borrowed from DCGAN tutorial (PyTorch).
    """
    def __init__(self, n_classes=10, latent_sz=32, ngf=32, nc=3, img_shape=[3, 32, 32]):
        super(GenConv, self).__init__()

        self.n_classes = n_classes
        self.latent_sz = latent_sz
        inp_dim = self.latent_sz + self.n_classes
        self.img_shape = img_shape

        self.label_emb = nn.Embedding(n_classes, n_classes)
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( inp_dim, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, nc, 4, 2, 1, bias=False),
            # state size. (nc) x 32 x 32
            nn.Tanh()
        )

    def get_inp(self, ys):
        u_vec = torch.normal(0, 1, (len(ys), self.latent_sz)).to(ys.device)
        y_vec = self.label_emb(ys)
        return torch.cat([u_vec, y_vec], -1)

    def forward(self, ys=None):
        inp = self.get_inp(ys)
        inp = inp.view(inp.size(0), inp.size(1), 1, 1)
        x_gen = self.model(inp)
        return x_gen


if __name__ == "__main__":
    # test conv generator
    G = Generator()
    print(G)
    ys = torch.randint(0, 10, (10,)).to(torch.device('cpu'))
    x_gen = G(ys)
    assert x_gen.shape == torch.Size([10, 3, 32, 32])

    # test linear generator
    G = GenLin()
    print(G)
    x_gen = G(ys)
    assert x_gen.shape == torch.Size([10, 3, 32, 32])

    # test conv generator
    G = GenConv()
    print(G)
    x_gen = G(ys)
    assert x_gen.shape == torch.Size([10, 3, 32, 32])
