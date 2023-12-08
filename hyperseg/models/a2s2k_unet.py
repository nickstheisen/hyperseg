#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

class A2S2KUNet(SemanticSegmentationModule):
    def __init__(self, 
            bilinear : bool = True,
            a2s2k_out_size : int = 24,
            batch_norm : bool = True,
            se_reduction : int = 2,
            **kwargs):
        super(A2S2KUNet, self).__init__(**kwargs)

        self.save_hyperparameters()

        self.bilinear = bilinear
        self.batch_norm = batch_norm
        self.a2s2k_out_size = a2s2k_out_size
        self.se_reduction = se_reduction

        # a2s2k layers
        self.conv1x1 = nn.Conv3d(
            in_channels=1,
            out_channels=self.a2s2k_out_size,
            kernel_size=(7,1,1),
            stride=(2,1,1),
            padding=0
        )
        self.bn1x1 = nn.Sequential(
            nn.BatchNorm3d(
                self.a2s2k_out_size, eps=0.001, momentum=0.1, affine=True
            ),
            nn.ReLU(inplace=True)
        )

        self.conv3x3 = nn.Conv3d(
            in_channels = 1,
            out_channels=self.a2s2k_out_size,
            kernel_size=(7,3,3),
            stride=(2,1,1),
            padding=(0,1,1)
        )
        self.bn3x3 = nn.Sequential(
            nn.BatchNorm3d(
                self.a2s2k_out_size, eps=0.001, momentum=0.1, affine=True
            ),
            nn.ReLU(inplace=True)
        )
        self.pool = nn.AdaptiveAvgPool3d(1)
        # TODO: Code from official repo does not match the formalization in paper
        # find out whats correct
        # TODO: reduce the feature channel dimension with 1DConv to get the original 
        # tensor shape (necessary to plug in front of UNet)
                
        # UNet layers
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
        

    def forward(self, x):
        # A2S2K-Module
        ## kernel split
        print(f"shape(x)={x.shape}")
        # additional dimension required for 3d-conv
        x = x.unsqueeze(dim=1)
        print(f"shape(x_unsqueezed)={x.shape}")
        x_spec = self.conv1x1(x)
        x_spec = self.bn1x1(x_spec).unsqueeze(dim=1)
        print(f"shape(x_spec)={x_spec.shape}")

        x_spat = self.conv3x3(x)
        x_spat = self.bn3x3(x_spat).unsqueeze(dim=1)
        print(f"shape(x_spat)={x_spat.shape}")
        
        ## Fusion
        x_fused = torch.cat([x_spec, x_spat], dim=1)
        x_fused = torch.sum(x_fused, dim=1) # elementwise addition of spectral + spatial features
        x_fused = self.pool(x_fused)
        x_fused = 
        print(f"shape(x_fused)={x_fused.shape}")
        
        # UNet
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
        return logits
