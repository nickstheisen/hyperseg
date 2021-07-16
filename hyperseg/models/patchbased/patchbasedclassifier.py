#!/usr/bin/env python

import torch
from torch import nn
import torchmetrics

from .patchbased import list_models, get_model

import pytorch_lightning as pl

class PatchbasedClassifier(pl.LightningModule):

    def __init__(self, 
            model_name: str,
            num_bands: int,
            num_classes: int,
            loss_name: str = 'cross_entropy',
            learning_rate: float = 1e-4,
            optimizer_name: str = 'SGD',
            momentum: float = 0.0,
    ):
        super(PatchbasedClassifier, self).__init__()
       
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
        if loss_name == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()

        # metrics
        self.accuracy_train = torchmetrics.Accuracy()
        self.accuracy_val = torchmetrics.Accuracy()
        self.accuracy_test = torchmetrics.Accuracy()

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
        acc = self.accuracy_train(prediction, labels.squeeze(dim=1))
        self.log('train_loss_step', loss)
        self.log('train_acc_step', acc)
        return loss

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.accuracy_train.compute())

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        prediction = self.forward(inputs)
        loss = self.criterion(prediction, labels.squeeze(dim=1))
        acc = self.accuracy_val(prediction, labels.squeeze(dim=1))
        self.log('val_loss_step', loss)
        self.log('val_acc_step', acc)

    def validation_epoch_end(self, outs):
        self.log('val_acc_epoch', self.accuracy_val.compute())

    def test_step(self, test_batch, batch_idx):
        inputs, labels = test_batch
        prediction = self.forward(inputs)
        loss = self.criterion(prediction, labels.squeeze(dim=1))
        acc = self.accuracy_test(prediction, labels.squeeze(dim=1))
        self.log('test_loss_step', loss)
        self.log('test_acc_step', acc)

    def test_epoch_end(self, outs):
        self.log('val_acc_epoch', self.accuracy_test.compute())
