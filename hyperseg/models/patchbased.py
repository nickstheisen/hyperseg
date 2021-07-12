#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvBN3d(nn.Module):
    def __init__(self,
            in_channels,
            kernel_size,
            stride,
            padding,
            num_kernels):
        super(ConvBN3d, self).__init__()

        self.conv = nn.Conv3d(
                in_channels=in_channels,
                out_channels=num_kernels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding)
        self.bn = nn.BatchNorm3d(
                num_kernels, eps=0.001, momentum=0.1,
                affine=True)

    def forward(self, X):
        X = self.conv(X)
        X = F.relu(self.bn(X))
        return X

class ConvBN2d(nn.Module):
    def __init__(self,
            in_channels,
            kernel_size,
            stride,
            padding,
            num_kernels):
        super(ConvBN2d, self).__init__()

        self.conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=num_kernels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding)
        self.bn = nn.BatchNorm2d(
                num_kernels, eps=0.001, momentum=0.1,
                affine=True)
    def forward(self, X):
        X = self.conv(X)
        X = F.relu(self.bn(X))
        return X

class SpectralResidualBlock(nn.Module):
    def __init__(self, in_channels, spectral_kernel_size, stride, num_kernels):
        super(SpectralResidualBlock, self).__init__()

        kernel_size = (spectral_kernel_size, 1, 1)
        padding = (spectral_kernel_size//2, 0, 0)

        self.convbn1x1_1 = ConvBN3d(
                in_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                num_kernels=num_kernels)

        self.convbn1x1_2 = ConvBN3d(
                in_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                num_kernels=num_kernels)

    def forward(self, X):
        identity_mapping = X
        X = self.convbn1x1_1(X)
        X = self.convbn1x1_2(X)
        X += identity_mapping
        return X

class SpatialResidualBlock(nn.Module):
    def __init__(self, in_channels, spatial_kernel_size, stride, num_kernels):
        super(SpatialResidualBlock, self).__init__()

        kernel_size = (spatial_kernel_size, spatial_kernel_size)
        padding = (spatial_kernel_size//2, spatial_kernel_size//2)

        self.convbn_1 = ConvBN2d(
                in_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                num_kernels=num_kernels)

        self.convbn_2 = ConvBN2d(
                in_channels=in_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                num_kernels=num_kernels)

    def forward(self, X):
        identity_mapping = X
        X = self.convbn_1(X)
        X = self.convbn_2(X)
        X += identity_mapping
        return X

class SSRN(nn.Module):
    def __init__(self, num_bands, num_classes, num_kernels=24, patch_size=7):
        super(SSRN, self).__init__()
        self.name = 'SSRN'
        self.num_kernels = num_kernels
        
        ## Spectral Feature Extraction
        self.spectral_input_conv = ConvBN3d(
                in_channels=1,
                kernel_size=(patch_size,1,1),
                stride=(2,1,1,),
                padding=0,
                num_kernels=num_kernels)

        self.spectral_resBlock_1 = SpectralResidualBlock(
                in_channels=num_kernels,
                spectral_kernel_size=7,
                stride=1,
                num_kernels=num_kernels)

        self.spectral_resBlock_2 = SpectralResidualBlock(
                in_channels=num_kernels,
                spectral_kernel_size=7,
                stride=1,
                num_kernels=num_kernels)
        
        spectral_depth=(num_bands - patch_size)//2+1
        self.spectral_output_conv = ConvBN3d(
                in_channels=num_kernels,
                kernel_size=(spectral_depth, 1, 1),
                stride=1,
                padding=0,
                num_kernels=128)

        ## Spatial Feature Extraction

        self.spatial_input_conv = ConvBN2d(
                in_channels=128,
                kernel_size=(3,3),
                stride=1,
                padding=0,
                num_kernels=num_kernels)

        self.spatial_resBlock_1 = SpatialResidualBlock(
                in_channels=num_kernels,
                spatial_kernel_size=3,
                stride=1,
                num_kernels=num_kernels)

        self.spatial_resBlock_2 = SpatialResidualBlock(
                in_channels=num_kernels,
                spatial_kernel_size=3,
                stride=1,
                num_kernels=num_kernels)

        ## Pooling and Classification
        self.avg_pool = torch.nn.AvgPool2d(
                # no padding leads with 3x3 kernel to reduction of resolution of 2 in each direction
                kernel_size=(patch_size-2, patch_size-2)
                )

        self.fc = torch.nn.Linear(num_kernels, num_classes)

    def forward(self, X):
        # extract spectral features
        X = self.spectral_input_conv(X)
        X = self.spectral_resBlock_1(X)
        X = self.spectral_resBlock_2(X)
        X = self.spectral_output_conv(X)
        X = torch.squeeze(X)
        
        # extract spatial features
        X = self.spatial_input_conv(X)
        X = self.spatial_resBlock_1(X)
        X = self.spatial_resBlock_2(X)
        
        # pooling and classification
        X = self.avg_pool(X)
        X = self.fc(X.view(-1, self.num_kernels))

        return X
