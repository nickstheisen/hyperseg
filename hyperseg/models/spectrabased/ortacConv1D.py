#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math
from ..semsegmodule import SemanticSegmentationModule

"""
Implementation of 1D-CNN based on Ortac & Ozcan (2021) by Nick Theisen (nicktheisen@uni-koblenz.de)

Ortac, G., & Ozcan, G. (2021). Comparative study of hyperspectral image classification by multidimensional Convolutional Neural Network approaches to improve accuracy. Expert Systems with Applications, 182, 115280. https://doi.org/10.1016/j.eswa.2021.115280
"""

""" Extended version of proposed 1D-CNN """
class OrtacConv1DExt(SemanticSegmentationModule):
    def __init__(self,
                kernel_size=3,
                pool_size=2,
                **kwargs):
        super(OrtacConv1DExt, self).__init__(**kwargs)
        
        self.kernel_size = kernel_size
        self.pool_size = pool_size
        self.padding = pool_size // 2

        self.save_hyperparameters()

        self.conv1 = nn.Conv1d(1,64, self.kernel_size, padding=self.padding)
        self.conv2 = nn.Conv1d(64, 128, self.kernel_size, padding=self.padding)
        self.conv3 = nn.Conv1d(128, 256, self.kernel_size, padding=self.padding)
        self.pool = nn.MaxPool1d(self.pool_size)
        self.n_features = self._get_final_flattened_size()
        self.fc1 = nn.Linear(self.n_features, 150)
        self.fc2 = nn.Linear(150, 300)
        self.fc3 = nn.Linear(300, self.n_classes)

    def _get_final_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1,1, self.n_channels)
            x = self.pool(self.conv1(x))
            x = self.pool(self.conv2(x))
            x = self.pool(self.conv3(x))
        return x.numel()

    def forward(self, image):
        # Input is NCHW (Batch,Channel,Height, Width), but 1D Convs expect 
        # NCL (Batch, Channel, Length), therefore we need to reshape images to a large batch of
        # spectra. Because of the naming it gets a little confusing but C=1, L=n_channels
        
        # reshape and store original dimensions
        n,c,h,w = image.shape
        x = image.permute(0,2,3,1)

        x = x.reshape(-1,1,c)
        
        # convolutional part
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        # flatten vectors
        x = x.view(-1, self.n_features)
       
        # MLP part
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x) # logits (CrossEntropyLoss includes softmax calculation )
        
        # reshapte to original batch shape
        x = x.reshape(n,h,w,self.n_classes)
        x = x.permute(0,3,1,2)
        return x

