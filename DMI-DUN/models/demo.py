import math
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
from itertools import repeat as rep
import torch.nn.functional as F
import collections.abc as container_abcs
from torch.nn.modules.module import Module
from torch import Tensor
from torch import nn, einsum
from einops import rearrange
from einops import repeat
from einops.layers.torch import Rearrange
import torchvision
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
#from .aggregation_zeropad import LocalConvolution
#from mxnet.gluon.block import HybridBlock
#from mxnet.gluon import nn as n
from imageio import imsave
import cv2

class MsDUF(nn.Module):
    def __init__(self, config):
        super(MsDUF, self).__init__()
        self.config = config
        self.phi_size = 32
        points = self.phi_size ** 2
        phi_init = np.random.normal(0.0, (1 / points) ** 0.5, size=(int(config.ratio * points), points))
        self.phi = nn.Parameter(torch.from_numpy(phi_init).float(), requires_grad=True)
        self.Q = nn.Parameter(torch.from_numpy(np.transpose(phi_init)).float(), requires_grad=True)

        self.num_layers = 8

        #self.weights1 = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.weights_Stem = nn.Parameter(torch.tensor(1.), requires_grad=True)
        self.etas_Stem = nn.Parameter(torch.tensor(0.1), requires_grad=True)
        self.weights = []
        self.etas = []
        for i in range(self.num_layers - 1):
            self.weights.append(nn.Parameter(torch.tensor(1.), requires_grad=True))
            self.etas.append(nn.Parameter(torch.tensor(0.1), requires_grad=True))


        self.MS = nn.ModuleList()
        for i in range(self.num_layers-1):
            self.MS.append(MSmodule(i+2))

        self.MS_Stem=MS_Stem()

    def forward(self, inputs):
        batch_size = inputs.size(0)
        #H = inputs.size(2)
        W = inputs.size(3)
        y = self.sampling(inputs, self.phi_size)
        recon = self.recon(y, self.phi_size, batch_size,W)
        return recon

    def sampling(self, inputs, init_block):
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=init_block, dim=3), dim=0)
        inputs = torch.cat(torch.split(inputs, split_size_or_sections=init_block, dim=2), dim=0)
        inputs = torch.reshape(inputs, [-1, init_block ** 2])
        inputs = torch.transpose(inputs, 0, 1)
        y = torch.matmul(self.phi, inputs)
        #noise_sigma = 3 / 255.0 * torch.randn_like(y)
        #y = y + noise_sigma
        return y

    def recon(self, y, init_block, batch_size,W):
        idx = int(W / init_block)
        recon = torch.matmul(self.Q, y)
        recon = recon - self.weights_Stem * torch.mm(torch.transpose(self.phi, 0, 1), (torch.mm(self.phi, recon) - y))
        recon = torch.reshape(torch.transpose(recon, 0, 1), [-1, 1, init_block, init_block])
        recon = torch.cat(torch.split(recon, split_size_or_sections=idx * batch_size, dim=0), dim=2)
        recon = torch.cat(torch.split(recon, split_size_or_sections=batch_size, dim=0), dim=3)
        out,f5,d1,d2,d3,d4 = self.MS_Stem(recon)
        recon = recon - self.etas_Stem * out

        recon = torch.cat(torch.split(recon, split_size_or_sections=init_block, dim=3), dim=0)
        recon = torch.cat(torch.split(recon, split_size_or_sections=init_block, dim=2), dim=0)
        recon = torch.reshape(recon, [-1, init_block ** 2])
        recon = torch.transpose(recon, 0, 1)

        for i in range(self.num_layers-1):
            recon = recon - self.weights[i] * torch.mm(torch.transpose(self.phi, 0, 1), (torch.mm(self.phi, recon) - y))
            recon = torch.reshape(torch.transpose(recon, 0, 1), [-1, 1, init_block, init_block])
            recon = torch.cat(torch.split(recon, split_size_or_sections=idx * batch_size, dim=0), dim=2)
            recon = torch.cat(torch.split(recon, split_size_or_sections=batch_size, dim=0), dim=3)
            out,f5,d1,d2,d3,d4 = self.MS[i](recon,f5,d1,d2,d3,d4)
            recon = recon - self.etas[i] * out

            recon = torch.cat(torch.split(recon, split_size_or_sections=init_block, dim=3), dim=0)
            recon = torch.cat(torch.split(recon, split_size_or_sections=init_block, dim=2), dim=0)
            recon = torch.reshape(recon, [-1, init_block ** 2])
            recon = torch.transpose(recon, 0, 1)

        recon = torch.reshape(torch.transpose(recon, 0, 1), [-1, 1, init_block, init_block])
        recon = torch.cat(torch.split(recon, split_size_or_sections=idx * batch_size, dim=0), dim=2)
        recon = torch.cat(torch.split(recon, split_size_or_sections=batch_size, dim=0), dim=3)
        return recon


class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class Fusion(nn.Module):
    def __init__(self, dim_in,stage,dim_out):
        super().__init__()

        self.fusion = nn.Sequential(nn.BatchNorm2d(dim_in * stage),
                                nn.GELU(),
                                nn.Conv2d(dim_in * stage,dim_in*4, kernel_size=1, stride=1, bias=False),
                                nn.BatchNorm2d(dim_in*4),
                                nn.GELU(),
                                nn.Conv2d(dim_in*4, dim_out, kernel_size=3, stride=1, padding=1,bias=False),
                               )

    def forward(self, x):
        out = self.fusion(x)
        return out


class ConvBlock(nn.Module):

    def __init__(self, in_channels):
        super().__init__()
        conv = []
        conv.append(Block(dim=in_channels))
        #conv_relu.append(Block(dim=in_channels))
        conv.append(Block(dim=in_channels))
        self.conv = nn.Sequential(*conv)

    def forward(self, x):
        out = self.conv(x)
        return out


class MSmodule(nn.Module):
    def __init__(self,stage):
        super().__init__()

        dim = 32
        self.left_conv_1 = nn.Sequential(nn.Conv2d(1, dim, kernel_size=3, stride=1,padding=1,bias=False),
                                nn.BatchNorm2d(dim),
                                nn.GELU(),
                                ConvBlock(dim),
                                SELayer(dim))

        self.downsample_layer1 = nn.Sequential(
                LayerNorm(dim, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dim, dim*2, kernel_size=2, stride=2),
                #SELayer(dim*2),
                ConvBlock(in_channels=dim * 2),
                SELayer(dim * 2)
            )

        self.downsample_layer2 = nn.Sequential(
            LayerNorm(dim * 2, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dim*2, dim * 4, kernel_size=2, stride=2),
            #SELayer(dim * 4),
            ConvBlock(in_channels=dim * 4),
            #ConvBlock(in_channels=dim * 4),
            #ConvBlock(in_channels=dim * 4),
            SELayer(dim * 4)
        )

        self.downsample_layer3 = nn.Sequential(
            LayerNorm(dim * 4, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dim*4, dim * 8, kernel_size=2, stride=2),
            #SELayer(dim * 8),
            ConvBlock(in_channels=dim * 8),
            SELayer(dim * 8)
        )

        self.downsample_layer4 = nn.Sequential(LayerNorm(dim*8, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dim*8, dim * 16, kernel_size=2, stride=2),
            #Block(dim=dim * 16),
            ConvBlock(in_channels=dim * 16),
            SELayer(dim * 16),
        )
        self.fusion0=Fusion(16*dim,stage,16*dim)

        self.deconv_1 = nn.Sequential(
                            LayerNorm(dim*16, eps=1e-6, data_format="channels_first"),
                            nn.ConvTranspose2d(in_channels=dim * 16, out_channels=dim * 8, kernel_size=2, stride=2),
                            SELayer(dim*8))
        self.right_conv_1 = nn.Sequential(
                                nn.BatchNorm2d(dim * 16),
                                nn.GELU(),
                                nn.Conv2d(in_channels=dim * 16, out_channels=dim * 8, kernel_size=3, padding=1, stride=1,bias=False),
                                ConvBlock(in_channels=dim * 8),
                                #ConvBlock(in_channels=dim * 8),
                                #ConvBlock(in_channels=dim * 8),
                                SELayer(dim * 8))

        self.fusion1 = Fusion(8*dim, stage,8*dim)

        self.deconv_2 =  nn.Sequential(LayerNorm(dim*8, eps=1e-6, data_format="channels_first"),
                                nn.ConvTranspose2d(in_channels=dim*8, out_channels=dim*4, kernel_size=2, stride=2),
                                SELayer(dim*4))
        self.right_conv_2 = nn.Sequential(
                                nn.BatchNorm2d(dim * 8),
                                nn.GELU(),
                                nn.Conv2d(in_channels=dim * 8, out_channels=dim * 4, kernel_size=3, padding=1, stride=1,bias=False),
                                ConvBlock(in_channels=dim * 4),
                                #ConvBlock(in_channels=dim * 4),
                                #ConvBlock(in_channels=dim * 4),
                                SELayer(dim * 4))

        self.fusion2 = Fusion(4*dim, stage,4*dim)

        self.deconv_3 = nn.Sequential(LayerNorm(dim*4, eps=1e-6, data_format="channels_first"),
                            nn.ConvTranspose2d(in_channels=dim*4, out_channels=dim*2, kernel_size=2, stride=2),
                            SELayer(dim*2))
        self.right_conv_3 = nn.Sequential(
                                nn.BatchNorm2d(dim * 4),
                                nn.GELU(),
                                nn.Conv2d(in_channels=dim * 4, out_channels=dim * 2, kernel_size=3, padding=1, stride=1,bias=False),
                                ConvBlock(in_channels=dim * 2),
                                SELayer(dim * 2))

        self.fusion3 = Fusion(dim * 2, stage, dim * 2)

        self.deconv_4 = nn.Sequential(LayerNorm(dim*2, eps=1e-6, data_format="channels_first"),
                            nn.ConvTranspose2d(in_channels=dim*2, out_channels=dim, kernel_size=2, stride=2),
                            SELayer(dim))
        self.right_conv_4 = nn.Sequential(
                                nn.BatchNorm2d(dim*2),
                                nn.GELU(),
                                nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=3, padding=1, stride=1,bias=False),
                                ConvBlock(in_channels=dim),
                                SELayer(dim))
        self.fusion4 = Fusion(dim, stage, dim)

        self.right_conv_5 = nn.Conv2d(in_channels=dim, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self,x,d0,d1,d2,d3,d4):
        # 1：Encoding
        feature_1 = self.left_conv_1(x)
        feature_2 = self.downsample_layer1(feature_1)
        feature_3 = self.downsample_layer2(feature_2)
        feature_4 = self.downsample_layer3(feature_3)
        feature_5 = self.downsample_layer4(feature_4)
        feature_5_stage = torch.cat((d0, feature_5), dim=1)
        feature_5 = self.fusion0(feature_5_stage)
        # 2：decoding
        de_feature_1 = self.deconv_1(feature_5)
        temp = torch.cat((feature_4, de_feature_1), dim=1)
        de_feature_1_conv = self.right_conv_1(temp)

        de_feature_1_stage = torch.cat((d1, de_feature_1_conv), dim=1)
        de_feature_1_conv = self.fusion1(de_feature_1_stage)

        de_feature_2 = self.deconv_2(de_feature_1_conv)
        temp = torch.cat((feature_3, de_feature_2), dim=1)

        de_feature_2_conv = self.right_conv_2(temp)

        de_feature_2_stage = torch.cat((d2, de_feature_2_conv), dim=1)
        de_feature_2_conv = self.fusion2(de_feature_2_stage)
        de_feature_3 = self.deconv_3(de_feature_2_conv)
        temp = torch.cat((feature_2, de_feature_3), dim=1)
        de_feature_3_conv = self.right_conv_3(temp)
        de_feature_3_stage = torch.cat((d3, de_feature_3_conv), dim=1)
        de_feature_3_conv = self.fusion3(de_feature_3_stage)

        de_feature_4 = self.deconv_4(de_feature_3_conv)

        temp = torch.cat((feature_1, de_feature_4), dim=1)
        de_feature_4_conv = self.right_conv_4(temp)
        de_feature_4_stage = torch.cat((d4, de_feature_4_conv), dim=1)
        de_feature_4_conv = self.fusion4(de_feature_4_stage)

        out = self.right_conv_5(de_feature_4_conv)
        return out,feature_5_stage,de_feature_1_stage,de_feature_2_stage,de_feature_3_stage,de_feature_4_stage

class MS_Stem(nn.Module):
    def __init__(self):
        super().__init__()

        dim = 32
        self.left_conv_1 = nn.Sequential(
                nn.Conv2d(1, dim, kernel_size=3, stride=1,padding=1),
                nn.BatchNorm2d(dim),
                nn.GELU(),
                ConvBlock(dim),
                SELayer(dim)
                #LayerNorm(dim*2, eps=1e-6, data_format="channels_first"),
            )

        self.downsample_layer1 = nn.Sequential(LayerNorm(dim, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dim, dim*2, kernel_size=2, stride=2),
                ConvBlock(in_channels=dim * 2),
                SELayer(dim * 2)
            )

        self.downsample_layer2 = nn.Sequential(
            LayerNorm(dim * 2, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dim*2, dim * 4, kernel_size=2, stride=2),
            #ConvBlock(in_channels=dim * 4),
            #ConvBlock(in_channels=dim * 4),
            ConvBlock(in_channels=dim * 4),
            SELayer(dim * 4)
        )

        self.downsample_layer3 = nn.Sequential(LayerNorm(dim*4, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dim*4, dim * 8, kernel_size=2, stride=2),
            ConvBlock(in_channels=dim * 8),
            SELayer(dim * 8)
        )


        self.downsample_layer4 = nn.Sequential(
            LayerNorm(dim * 8, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dim*8, dim * 16, kernel_size=2, stride=2),
            #Block(dim=dim * 16),
            ConvBlock(in_channels=dim * 16),
            SELayer(dim * 16)
        )


        self.deconv_1 = nn.Sequential(LayerNorm(dim*16, eps=1e-6, data_format="channels_first"),
                                      nn.ConvTranspose2d(in_channels=dim * 16, out_channels=dim * 8, kernel_size=2, stride=2),
                                      SELayer(dim * 8))
        self.right_conv_1 = nn.Sequential(
                                nn.BatchNorm2d(dim *16),
                                nn.GELU(),
                                nn.Conv2d(in_channels=dim * 16, out_channels=dim * 8, kernel_size=3, padding=1, stride=1,bias=False),
                                ConvBlock(in_channels=dim * 8),
                                #ConvBlock(in_channels=dim * 8),
                                #ConvBlock(in_channels=dim * 8),
                                SELayer(dim * 8)
        )

        self.deconv_2 = nn.Sequential(LayerNorm(dim*8, eps=1e-6, data_format="channels_first"),
                            nn.ConvTranspose2d(in_channels=dim*8, out_channels=dim*4, kernel_size=2, stride=2),
                            SELayer(dim * 4))
        self.right_conv_2 = nn.Sequential(
                                nn.BatchNorm2d(dim * 8),
                                nn.GELU(),
                                nn.Conv2d(in_channels=dim * 8, out_channels=dim * 4, kernel_size=3, padding=1, stride=1,bias=False),
                                ConvBlock(in_channels=dim * 4),
                                #ConvBlock(in_channels=dim * 4),
                                #ConvBlock(in_channels=dim * 4),
                                SELayer(dim * 4))

        self.deconv_3 = nn.Sequential(LayerNorm(dim*4, eps=1e-6, data_format="channels_first"),
                            nn.ConvTranspose2d(in_channels=dim*4, out_channels=dim*2, kernel_size=2, stride=2),
                            SELayer(dim * 2))
        self.right_conv_3 = nn.Sequential(
                                nn.BatchNorm2d(dim * 4),
                                nn.GELU(),
                                nn.Conv2d(in_channels=dim * 4, out_channels=dim * 2, kernel_size=3, padding=1, stride=1),
                                ConvBlock(in_channels=dim * 2),
                                SELayer(dim * 2))

        self.deconv_4 = nn.Sequential(LayerNorm(dim*2, eps=1e-6, data_format="channels_first"),
                            nn.ConvTranspose2d(in_channels=dim*2, out_channels=dim, kernel_size=2, stride=2),
                            SELayer(dim))
        self.right_conv_4 = nn.Sequential(
                                nn.BatchNorm2d(dim*2),
                                nn.GELU(),
                                nn.Conv2d(in_channels=dim * 2, out_channels=dim, kernel_size=3, padding=1, stride=1),
                                ConvBlock(in_channels=dim),
                                SELayer(dim))

        self.right_conv_5 = nn.Conv2d(in_channels=dim, out_channels=1, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # 1：Encoding
        feature_1 = self.left_conv_1(x)
        feature_2 = self.downsample_layer1(feature_1)
        feature_3 = self.downsample_layer2(feature_2)
        feature_4 = self.downsample_layer3(feature_3)
        feature_5 = self.downsample_layer4(feature_4)
        # 2：decoding
        de_feature_1 = self.deconv_1(feature_5)
        temp = torch.cat((feature_4, de_feature_1), dim=1)
        de_feature_1_conv = self.right_conv_1(temp)

        de_feature_2 = self.deconv_2(de_feature_1_conv)
        temp = torch.cat((feature_3, de_feature_2), dim=1)
        de_feature_2_conv = self.right_conv_2(temp)
        de_feature_3 = self.deconv_3(de_feature_2_conv)
        temp = torch.cat((feature_2, de_feature_3), dim=1)
        de_feature_3_conv = self.right_conv_3(temp)
        de_feature_4 = self.deconv_4(de_feature_3_conv)
        temp = torch.cat((feature_1, de_feature_4), dim=1)
        de_feature_4_conv = self.right_conv_4(temp)

        out = self.right_conv_5(de_feature_4_conv)
        return out,feature_5,de_feature_1_conv,de_feature_2_conv,de_feature_3_conv,de_feature_4_conv


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class Block(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x

