#!/usr/bin/env python
# coding: utf-8

'''
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from .semsegmodule import SemanticSegmentationModule
import hyperseg.models.network as network

class DeeplabV3Plus(SemanticSegmentationModule):
    def __init__(self,
            backbone: str = 'resnet101',
            pretrained_weights:str = None,
            pretrained_backbone:bool = False,
            **kwargs):
        super(DeeplabV3Plus, self).__init__(**kwargs)

        self.save_hyperparameters()
        self.pt_weights = pretrained_weights
        self.model = network.modeling.__dict__[f'deeplabv3plus_{backbone}'](
                        num_classes = self.n_classes,
                        output_stride=16,
                        pretrained_backbone=pretrained_backbone,
                        num_channels = self.n_channels)
        
        if not (self.pt_weights is None):
            model_state = torch.load( self.pt_weights )['model_state']
            ## replace output layer to match n_classes
            weight_shape = model_state['classifier.classifier.3.weight'].shape
            bias_shape = model_state['classifier.classifier.3.bias'].shape
            # update classifier weight shapes
            weight_shape = torch.Size([self.n_classes, weight_shape[1], 
                                            weight_shape[2], weight_shape[3]])
            bias_shape = torch.Size([self.n_classes])
            # insert random values
            model_state['classifier.classifier.3.weight'] = torch.rand(weight_shape)
            model_state['classifier.classifier.3.bias'] = torch.rand(bias_shape)

            self.model.load_state_dict( model_state )
            
            # freeze everything except classifier
            for param in self.model.parameters():
                param.requires_grad = False
            
            for param in self.model.classifier.parameters():
                param.requires_grad = True

    def forward(self, x):
        x = self.model(x)
        return x

