#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .imagebased import SemanticSegmentationModule

import kornia as K
from kornia.augmentation import AugmentationSequential

def ceildiv(a, b):
    # usage of ceil division instead of floor div (//) to avoid 0 for 1 // 2
    return -(a // -b)

#class ModLayer(nn.Module):
#    def __init__(self, N, C, H, W):
#        super().__init__()
#        self.N = N
#        self.C = C
#        self.H = H
#        self.W = W
#        
#
#        self.a_cnn = nn.Sequential(
#                        nn.Conv2d(256, 512, kernel_size=3, padding=1),
#                        nn.ReLU(inplace=True),
#                        nn.Conv2d(512, 1024, kernel_size=3, padding=1),
#                        nn.ReLU(inplace=True))
#        
#        self.b_cnn = nn.Sequential(
#                        nn.Conv2d(256, 512, kernel_size=3, padding=1),
#                        nn.ReLU(inplace=True),
#                        nn.Conv2d(512, 1024, kernel_size=3, padding=1),
#                        nn.ReLU(inplace=True))
#        
#        self.dim_red_a = nn.Conv2d(1024, 16, kernel_size=1)
#        self.dim_red_b = nn.Conv2d(1024, 16, kernel_size=1)
#         
#    def forward(self, x):
#        #sharpen1 = K.filters.UnsharpMask((5,5), (self.sharpen_sigma1, self.sharpen_sigma1))
#        # [N, C, H, W]
#        #  0  1  2  3
#        # reshape to [N, W, H, C]
#        #             0  3  2  1
#        # reshape to [N, H, C, W]
#        #             0  2  1  3
#        # need to get the current H/W for some reason, appears to be swapped between train/val/sanity_check???
#        self.W = x.size()[3]
#        self.H = x.size()[2]
#
#        
#        x = F.interpolate(x, size=(256, 256), mode="bilinear")
#        a = torch.permute(x, (0,3,2,1))
#        b = torch.permute(x, (0,2,1,3)) 
#        
#
#        x1 = self.a_cnn(a)
#        x2 = self.b_cnn(b)
#
#        #a size [32, 1024, 128, N_channels/2]
#        #b size [32, 1024, N_channels/2, 128]
#        x1 = self.dim_red_a(x1)
#        x2 = self.dim_red_b(x2)
#        x1 = F.interpolate(x1, size=(self.H,self.W), mode="bilinear")
#        x2 = F.interpolate(x2, size=(self.H,self.W), mode="bilinear")
#        x = torch.cat((x1,x2), 1)
#        return x
        
class ModLayer(nn.Module):
    def __init__(self):
        super().__init__()
        # learnable sharpening
        self.sharpen_sigma1 = nn.Parameter(torch.tensor(5.0), requires_grad=True)
        
    def forward(self, x):
        print(self.sharpen_sigma1.data)
        sharpen = K.filters.UnsharpMask((5,5), (self.sharpen_sigma1, self.sharpen_sigma1))
        x = sharpen(x)
        return x

class AugmentationModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.augmentation = AugmentationSequential(
                                K.augmentation.RandomHorizontalFlip(p=0.5),
                                data_keys=["input","mask"])
        
    def forward(self, x, labels):
        shape = x.shape
        labels = torch.reshape(labels,(shape[0], 1, shape[2], shape[3]))
        augmented = self.augmentation(x, labels.float())
        #print(augmented[0].size())
        #print(augmented[1].size())
        
        return augmented[0], augmented[1]


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, batch_norm=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if batch_norm:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, batch_norm=True, dropout=0.0):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Dropout(dropout),
            DoubleConv(in_channels, out_channels, batch_norm=batch_norm)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True, batch_norm=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, batch_norm=batch_norm)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, batch_norm=batch_norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNetAug(SemanticSegmentationModule):
    def __init__(self, 
            #batch_size : int,
            #height : int,
            #width : int,
            bilinear : bool = True,
            batch_norm : bool = True,
            dropout : float = 0.0,
            **kwargs):
        super(UNetAug, self).__init__(**kwargs)

        self.save_hyperparameters()

        self.bilinear = bilinear
        self.batch_norm = batch_norm
        self.dropout = dropout
        # need to know N, H , W at init time for modlayer
        #self.H = height
        #self.W = width
        #self.N = batch_size
        #self.C = self.n_channels

        #self.mod = ModLayer(self.n_channels, ceildiv(self.n_channels,4), dropout2d=self.dropout2d)
        #self.inc = DoubleConv(ceildiv(self.n_channels,4), 64, batch_norm=self.batch_norm)
        self.mod = ModLayer()
        if self.augmentation:
            self.aug = AugmentationModule()
        self.inc = DoubleConv(self.n_channels, 64, batch_norm=self.batch_norm)
        self.down1 = Down(64, 128, batch_norm=self.batch_norm)
        self.down2 = Down(128, 256, batch_norm=self.batch_norm)
        self.down3 = Down(256, 512, batch_norm=self.batch_norm)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor, batch_norm=self.batch_norm)
        self.up1 = Up(1024, 512 // factor, bilinear, batch_norm=self.batch_norm)
        self.up2 = Up(512, 256 // factor, bilinear, batch_norm=self.batch_norm)
        self.up3 = Up(256, 128 // factor, bilinear, batch_norm=self.batch_norm)
        self.up4 = Up(128, 64, bilinear, batch_norm=self.batch_norm)
        self.outc = OutConv(64, self.n_classes)

    def forward(self, x, labels=None, val_mode=False):
        # TODO inc below isnt taking previous as input!!!
        #x = torch.cat((x,m),1)
        x = self.mod(x)
        if not val_mode:
            x, labels = self.aug(x, labels)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        if self.augmentation:
            return logits, labels
        else:
            return logits
