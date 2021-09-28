#!/usr/bin/env python

import torch
from torch import nn
import torchmetrics

from .unet import UNet

import pytorch_lightning as pl

def list_models():
    model_list = ['UNet']
    return model_list

def get_model(modelname, num_bands, num_classes):
    model_list = list_models()
    if modelname in model_list:
        if modelname == 'UNet':
            return UNet(num_bands, num_classes)

class ImagebasedClassifier(pl.LightningModule):

    def __init__(self, 
            model_name: str,
            num_bands: int,
            num_classes: int,
            loss_name: str = 'cross_entropy',
            learning_rate: float = 1e-4,
            optimizer_name: str = 'SGD',
            momentum: float = 0.0,
            ignore_index: int = 0,
    ):
        super(ImagebasedClassifier, self).__init__()
       
        # initialize dataset
        self.num_bands = num_bands
        self.num_classes = num_classes
        
        # initialize model
        self.model_name = model_name
        self.model = get_model(model_name, self.num_bands, self.num_classes)

        # optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.optimizer_name = optimizer_name

        # loss
        self.ignore_index = ignore_index
        if loss_name == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # metrics
        self.accuracy_train = torchmetrics.Accuracy(ignore_index=self.ignore_index)
        self.accuracy_val = torchmetrics.Accuracy(ignore_index=self.ignore_index)
        self.accuracy_test = torchmetrics.Accuracy(ignore_index=self.ignore_index)

    def forward(self, x):
        res = self.model(x)
        return res

    def configure_optimizers(self):
        if self.optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), 
                    lr=self.learning_rate, 
                    momentum=self.momentum)
        elif self.optimizer_name == 'RMSprop':
            optimizer = torch.optim.RMSprop(self.parameters(),
                    lr=self.learning_rate,
                    momentum=self.momentum)
        elif self.optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(),
                    lr=self.learning_rate,
                    momentum=self.momentum)
        else :
            raise RuntimeError(f'Optimizer {self.optimizer_name} unknown!')

        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        prediction = self.forward(inputs)
        loss = self.criterion(prediction, labels.squeeze(dim=1))
        acc = self.accuracy_train(prediction.argmax(dim=1), labels.squeeze(dim=1))
        self.log('train_loss_step', loss)
        self.log('train_acc_step', acc)
        return loss

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.accuracy_train.compute())

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        prediction = self.forward(inputs)
        loss = self.criterion(prediction, labels.squeeze(dim=1))
        acc = self.accuracy_val(prediction.argmax(dim=1), labels.squeeze(dim=1))
        self.log('val_loss_step', loss)
        self.log('val_acc_step', acc)

    def validation_epoch_end(self, outs):
        self.log('val_acc_epoch', self.accuracy_val.compute())

    def test_step(self, test_batch, batch_idx):
        inputs, labels = test_batch
        prediction = self.forward(inputs)
        loss = self.criterion(prediction, labels.squeeze(dim=1))
        acc = self.accuracy_test(prediction.argmax(dim=1), labels.squeeze(dim=1))
        self.log('test_loss_step', loss)
        self.log('test_acc_step', acc)

    def test_epoch_end(self, outs):
        self.log('val_acc_epoch', self.accuracy_test.compute())
