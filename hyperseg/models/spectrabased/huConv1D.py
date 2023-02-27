#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from ..semsegmodule import SemanticSegmentationModule

'''
Implementation of Model proposed in
W. Hu, Y. Huang, L. Wei, F. Zhang, and H. Li, “Deep Convolutional Neural Networks for Hyperspectral Image Classification,” Journal of Sensors, vol. 2015, p. e258619, Jul. 2015, doi: 10.1155/2015/258619.

Inspired by implementation in https://gitlab.inria.fr/naudeber/DeepHyperX/-/blob/master/models.py
'''
class HuConv1D(SemanticSegmentationModule):
    def __init__(self,
                kernel_size=None,
                pool_size=None,
                **kwargs):
        super(HuConv1D, self).__init__(**kwargs)
        if kernel_size is None:
            self.kernel_size = math.ceil(self.n_channels/5)
        else:
            self.kernel_size = kernel_size
        if pool_size is None:
            self.pool_size = 2
        else:
            self.pool_size = pool_size

        self.save_hyperparameters()

        self.conv = nn.Conv1d(1, 20, self.kernel_size)
        self.pool = nn.MaxPool1d(pool_size)
        self.n_features = self._get_final_flattened_size()
        self.fc1 = nn.Linear(self.n_features, 100)
        self.fc2 = nn.Linear(100, self.n_classes)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, self.n_channels)
            x = self.pool(self.conv(x))
        return x.numel()

    def forward(self, image):
        # Input is NCHW (Batch,Channel,Height, Width), but 1D Convs expect 
        # NCL (Batch, Channel, Length), therefore we need to reshape images to a large batch of
        # spectra. Because of the naming it gets a little confusing but C=1, L=n_channels
        
        # reshape and store original dimensions
        n,c,h,w = image.shape
        x = image.permute(0,2,3,1)

        x = x.reshape(-1,1,c)
        # convolutional part of network (we use tanh as in DeepHyperX) 
        x = torch.tanh(self.pool(self.conv(x)))
        
        # reshape all 20 features maps to one long feature vector
        x = x.view(-1, self.n_features)
        
        # fully connected part of NN
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)

        # reshape back to original image shape
        x = x.reshape(n,h,w,self.n_classes)
        x = x.permute(0,3,1,2)

        return x

