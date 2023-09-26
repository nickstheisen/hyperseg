#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ..semsegmodule import SemanticSegmentationModule

import matplotlib.pyplot as plt

DEBUG = False

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

    def __init__(self, in_channels, out_channels, batch_norm=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
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

class AttentionGate(nn.Module):
    def __init__(self, in_channels, gate_channels):
        super(AttentionGate, self).__init__()
        inter_channels = in_channels # half the gate size
        self.conv1x1_g = nn.Conv2d(gate_channels, inter_channels, kernel_size=1)
        self.conv1x1_x = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        
        self.downsample = nn.Conv2d(inter_channels, inter_channels, kernel_size=3, stride=2, 
                              padding=1)
        self.conv1x1_psi = nn.Conv2d(inter_channels, 1, kernel_size=1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x, g):
        # pad to handle problems with odd image resolutions
        # x.shape and g.shape*2 can differ in x/y-directions by 1px, with shape(g)<shape(x)
        # Happens when shape(x)% 2 != 0 -> 1px is lost 
        diffY = x.shape[2] - g.shape[2]*2
        diffX = x.shape[3] - g.shape[3]*2
        if DEBUG:
            print(f"x: {x.shape}\tg: {g.shape}")
            print(f"diffY = {diffY}\tdiffX = {diffX}")
        
        # if a difference exists, both images need to be padded
        # example: x.shape = (N, C, 27, 51), g.shape = (N, C, 13, 25)
        # if only x is padded after downsampling we get (N, C, 14, 26) which is incompatible
        # with g. Therefore g needs to be padded as well to (N,C, 14, 26)
        # After upscaling we can then crop to the original size to make it compatible with input x
        # again.
        # Zero padding after upscaling is not optimal because then a whole row (and column) just
        # has zero attention which basically removes information.
        x_pad = F.pad(x, [0, diffX, 0, diffY])
        g_pad = F.pad(g, [0, diffX, 0, diffY])

        # calculate gating coefficient (attention map)
        g1 = self.conv1x1_g(g_pad) # convolve gating signal vector
        x1 = self.conv1x1_x(x_pad) # convolve feature map
        x_ds = self.downsample(x1)
        if DEBUG:
            print(f"shape(x1): {x1.shape}")
            print(f"shape(g1): {g1.shape}")
            print(f"shape(x_ds): {x_ds.shape}")

        x_sum = F.relu(torch.add(x_ds, g1))
        if DEBUG:
            print(f"shape(x_sum): {x_sum.shape}")
        x_sum = F.sigmoid(self.conv1x1_psi(x_sum))
        att_coeff = self.upsample(x_sum) # attention coefficient

        # crop to original dimensionality
        att_coeff = att_coeff[:, :, 0:att_coeff.shape[2] - diffY, 0:att_coeff.shape[3] - diffX]
        if DEBUG:
            plt.matshow(att_coeff[0].detach().cpu().squeeze())
            plt.show()
            print(f"shape(att_coeff): {att_coeff.shape}")
        x_att = torch.mul(x, att_coeff)
        if DEBUG:
            print(f"shape(x_att): {x_att.shape}")
        
        return x_att, att_coeff
        

class AGUNet(SemanticSegmentationModule):
    def __init__(self, 
            bilinear : bool = True,
            dim_reduction  : int = None,
            batch_norm : bool = True,
            model_name: str = "agunet",
            **kwargs):
        super(AGUNet, self).__init__(**kwargs)

        self.save_hyperparameters()
        self.filters = [64, 128, 256, 512, 1024]

        self.bilinear = bilinear
        self.dr = dim_reduction
        self.batch_norm = batch_norm

        if self.dr is not None:
            self.dr_layer = torch.nn.Conv2d(self.n_channels, self.dr, 1)
            self.inc = DoubleConv(self.dr, 64, batch_norm=batch_norm)
        else :
            self.inc = DoubleConv(self.n_channels, 64, batch_norm=self.batch_norm)
        self.down1 = Down(self.filters[0], self.filters[1], batch_norm=self.batch_norm)
        self.down2 = Down(self.filters[1], self.filters[2], batch_norm=self.batch_norm)
        self.down3 = Down(self.filters[2], self.filters[3], batch_norm=self.batch_norm)
        factor = 2 if bilinear else 1
        self.down4 = Down(self.filters[3], self.filters[4] // factor, batch_norm=self.batch_norm)
        self.up1 = Up(self.filters[4], self.filters[3] // factor, bilinear, 
            batch_norm=self.batch_norm)
        self.up2 = Up(self.filters[3], self.filters[2] // factor, bilinear, 
            batch_norm=self.batch_norm)
        self.up3 = Up(self.filters[2], self.filters[1] // factor, bilinear, 
            batch_norm=self.batch_norm)
        self.up4 = Up(self.filters[1], self.filters[0], bilinear, batch_norm=self.batch_norm)
        self.outc = OutConv(self.filters[0], self.n_classes)

        self.ag1 = AttentionGate(in_channels=self.filters[0],
                                 gate_channels=self.filters[1] // factor)
        self.ag2 = AttentionGate(in_channels=self.filters[1],
                                 gate_channels=self.filters[2] // factor)
        self.ag3 = AttentionGate(in_channels=self.filters[2],
                                 gate_channels=self.filters[3] // factor)
        self.ag4 = AttentionGate(in_channels=self.filters[3],
                                 gate_channels=self.filters[4] // factor)
        

    def forward(self, x):
        if self.dr is not None:
            x = self.dr_layer(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        if DEBUG:
            print(f"x: {x.shape}")
            print(f"x1: {x1.shape}")
            print(f"x2: {x2.shape}")
            print(f"x3: {x3.shape}")
            print(f"x4: {x4.shape}")
            print(f"x5: {x5.shape}")
        
        x4_att, att4 = self.ag4(x4, x5) #x5 is gating signal
        x = self.up1(x5, x4_att) # use attention gated x4 instead of x4 (skip conn.)
        x3_att, att3 = self.ag3(x3, x) 
        if DEBUG:
            print(f"x4_att: {x4_att.shape}")
            print(f"x: {x.shape}")
            print(f"x3_att: {x3_att.shape}")
        x = self.up2(x, x3_att)
        x2_att, att2 = self.ag2(x2, x)
        if DEBUG:
            print(f"x: {x.shape}")
            print(f"x2_att: {x2_att.shape}")
        x = self.up3(x, x2_att)
        x1_att, att1 = self.ag1(x1, x)
        if DEBUG:
            print(f"x: {x.shape}")
            print(f"x1_att: {x1_att.shape}")
        x = self.up4(x, x1_att)
        if DEBUG:
            print(f"x: {x.shape}")
        logits = self.outc(x)
        self.last_att_coeffs = [att4, att3, att2, att1]
        return logits
