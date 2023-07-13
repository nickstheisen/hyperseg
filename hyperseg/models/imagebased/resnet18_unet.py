#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import models

from ..semsegmodule import SemanticSegmentationModule

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

class ResNet18UNet(SemanticSegmentationModule):
    def __init__(self, 
            bilinear : bool = True,
            dim_reduction  : int = None,
            batch_norm : bool = True,
            **kwargs):
        super(ResNet18UNet, self).__init__(**kwargs)

        self.save_hyperparameters()

        self.resnet18 = models.resnet18()
        self.resnet18_layers = list(self.resnet18.children())

        self.bilinear = bilinear
        self.dr = dim_reduction
        self.batch_norm = batch_norm

        self.down1 = nn.Sequential(*self.resnet18_layers[3:5]) 
        self.down2 = nn.Sequential(*self.resnet18_layers[5])
        self.down3 = nn.Sequential(*self.resnet18_layers[6])
        self.down4 = nn.Sequential(*self.resnet18_layers[7])

        if self.dr is not None:
            self.dr_layer = torch.nn.Conv2d(self.n_channels, self.dr, 1)
            self.inc = DoubleConv(self.dr, 64, batch_norm=batch_norm)
        else :
            self.inc = DoubleConv(self.n_channels, 64, batch_norm=self.batch_norm)
        factor = 2 if bilinear else 1
        self.up1 = Up(512 + 256, 512 // factor, bilinear, batch_norm=self.batch_norm)
        self.up2 = Up(256 + 128, 256 // factor, bilinear, batch_norm=self.batch_norm)
        self.up3 = Up(128 + 64, 128 // factor, bilinear, batch_norm=self.batch_norm)
        self.up4 = Up(64 + 64, 64, bilinear, batch_norm=self.batch_norm)
        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        if self.dr is not None:
            x = self.dr_layer(x)
        x1 = self.inc(x)
        x2 = self.down1(x1) # x2 has 64 channels
        x3 = self.down2(x2) # x3 has 128 channels
        x4 = self.down3(x3) # x4 has 256 channels
        x5 = self.down4(x4) # x5 has 512 channels
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
