#!/usr/bin/env python

import torch
from torch import nn
import torchmetrics

import torchvision.transforms as T

import pytorch_lightning as pl

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.switch_backend('agg')

class SemanticSegmentationClassifier(pl.LightningModule):
    def __init__(self, 
            n_channels: int,
            n_classes: int,
            label_def: str,
            loss_name: str,
            learning_rate: float,
            optimizer_name: str,
            momentum: float,
            ignore_index: int,
            mdmc_average: str,
    ):
        super(SemanticSegmentationClassifier, self).__init__()
        self.save_hyperparameters()

        self.num_labels = n_classes
        self.n_channels = n_channels
       
        # optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.optimizer_name = optimizer_name

        # loss
        self.ignore_index = ignore_index
        if loss_name == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # class definitions
        self.label_names, self.label_colors = self._load_label_def(label_def)

        # metrics
        self.mdmc_average = mdmc_average
        self.accuracy_train = torchmetrics.Accuracy(ignore_index=self.ignore_index, 
                mdmc_average=self.mdmc_average)
        self.accuracy_val = torchmetrics.Accuracy(ignore_index=self.ignore_index,
                mdmc_average=self.mdmc_average)
        self.accuracy_test = torchmetrics.Accuracy(ignore_index=self.ignore_index,
                mdmc_average=self.mdmc_average)
        self.f1_train = torchmetrics.F1(ignore_index=self.ignore_index,
                mdmc_average=self.mdmc_average)
        self.f1_val = torchmetrics.F1(ignore_index=self.ignore_index, 
                mdmc_average=self.mdmc_average)
        self.f1_test = torchmetrics.F1(ignore_index=self.ignore_index,
                mdmc_average=self.mdmc_average)
        self.iou_train = torchmetrics.IoU(ignore_index=self.ignore_index, 
                num_classes=self.num_labels+1) # number of labels + undefined
        self.iou_val = torchmetrics.IoU(ignore_index=self.ignore_index, 
                num_classes=self.num_labels+1) # number of labels + undefined
        self.iou_test = torchmetrics.IoU(ignore_index=self.ignore_index, 
                num_classes=self.num_labels+1) # number of labels + undefined
        self.confmat_train = torchmetrics.ConfusionMatrix(
                num_classes=self.num_labels+1) # number of labels + undefined
        self.confmat_val=torchmetrics.ConfusionMatrix(
                num_classes=self.num_labels+1) # number of labels + undefined
        self.confmat_test = torchmetrics.ConfusionMatrix(
                num_classes=self.num_labels+1) # number of labels + undefined

    # pretrain_routine hook
    def on_pretrain_routine_start(self):
        # logging
        self.confmat_log_dir = Path(self.logger.log_dir).joinpath('confmats')
        self.confmat_log_dir.mkdir(parents=True, exist_ok=True)
    
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
                    lr=self.learning_rate)
        else :
            raise RuntimeError(f'Optimizer {self.optimizer_name} unknown!')

        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        prediction = self.forward(inputs)

        prediction =  T.Resize((labels.shape[1:3]))(prediction)
        loss = self.criterion(prediction, labels.squeeze(dim=1))

        prediction = prediction.argmax(dim=1, keepdims=True)
        labels = labels.unsqueeze(1)
        self.accuracy_train(prediction, labels)
        self.f1_train(prediction, labels)
        self.iou_train(prediction, labels)
        self.confmat_train(prediction, labels)
        self.log('train_loss_step', loss)
        return loss

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.accuracy_train.compute())
        self.log('train_f1_epoch', self.f1_train.compute())
        self.log('train_iou_epoch', self.iou_train.compute())
        confmat_epoch = self.confmat_train.compute().detach().cpu().numpy().astype(np.int)
        confmat_logpath = self.confmat_log_dir.joinpath(f'confmat_train_epoch{self.current_epoch}.csv')
        self._export_confmat(confmat_logpath, confmat_epoch)
        self.logger.experiment.add_figure("Confusion Matrix (train)", 
                self._plot_confmat(confmat_epoch),
                self.current_epoch)

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        prediction = self.forward(inputs)

        prediction =  T.Resize((labels.shape[1:3]))(prediction)
        loss = self.criterion(prediction, labels.squeeze(dim=1))

        prediction = prediction.argmax(dim=1, keepdims=True)
        labels = labels.unsqueeze(1)
        self.accuracy_val(prediction, labels)
        self.f1_val(prediction, labels)
        self.iou_val(prediction, labels)
        self.confmat_val(prediction, labels)
        self.log('val_loss_step', loss)
        if batch_idx == 0:
            self.logger.experiment.add_figure("Prediction (sample batch):",
                    self._plot_batch_prediction(prediction.detach().cpu().numpy().astype(np.int),
                        labels.detach().cpu().numpy().astype(np.int)),
                    self.current_epoch)


    def validation_epoch_end(self, outs):
        self.log('val_acc_epoch', self.accuracy_val.compute())
        self.log('val_f1_epoch', self.f1_val.compute())
        self.log('val_iou_epoch', self.iou_val.compute())
        confmat_epoch = self.confmat_val.compute().detach().cpu().numpy().astype(np.int)
        confmat_logpath = self.confmat_log_dir.joinpath(f'confmat_val_epoch{self.current_epoch}.csv')
        self._export_confmat(confmat_logpath, confmat_epoch)
        self.logger.experiment.add_figure("Confusion Matrix (validation)", 
                self._plot_confmat(confmat_epoch),
                self.current_epoch)

    def test_step(self, test_batch, batch_idx):
        inputs, labels = test_batch
        prediction = self.forward(inputs)

        prediction =  T.Resize((labels.shape[1:3]))(prediction)
        loss = self.criterion(prediction, labels.squeeze(dim=1))

        prediction = prediction.argmax(dim=1, keepdims=True)
        labels = labels.unsqueeze(1)
        self.accuracy_test(prediction, labels)
        self.f1_test(prediction, labels)
        self.iou_test(prediction, labels)
        self.confmat_test(prediction, labels)
        self.log('test_loss_step', loss)

    def test_epoch_end(self, outs):
        self.log('test_acc_epoch', self.accuracy_test.compute())
        self.log('test_f1_epoch', self.accuracy_test.compute())
        self.log('test_iou_epoch', self.accuracy_test.compute())
        confmat_epoch = self.confmat_test.compute().detach().cpu().numpy().astype(np.int)
        confmat_logpath = self.confmat_log_dir.joinpath(f'confmat_test_epoch{self.current_epoch}.csv')
        self._export_confmat(confmat_logpath, confmat_epoch)
        self.logger.experiment.add_figure("Confusion Matrix (test)", 
                self._plot_confmat(confmat_epoch),
                self.current_epoch)

    def _export_confmat(self, path, confmat):
        np.savetxt(path, confmat)

    def _plot_confmat(self, confmat):
        figure = plt.figure()
        plt.matshow(confmat[:-1,:-1], fignum=0)
        return figure

    def _load_label_def(self, label_def):
        label_defs = np.loadtxt(label_def, delimiter=',', dtype=str)
        label_names = np.array(label_defs[:,1])
        label_colors = np.array(label_defs[:, 2:], dtype='int')
        return label_names, label_colors

    def _plot_batch_prediction(self, pred, label):
        batch_size = label.shape[0]
        figure, axes = plt.subplots(nrows=2, ncols=batch_size, squeeze=False)
        for i in range(batch_size):
            axes[0, i].imshow(self.label_colors[label[i]].squeeze())
            axes[1, i].imshow(self.label_colors[pred[i]].squeeze())

        # add legend that shows label color class mapping
        handles = []
        for i, color in enumerate(self.label_colors):
            handles.append(mpatches.Patch(color=color*(1./255), label=self.label_names[i]))

        plt.legend(handles=handles, loc='center left', ncol=2, bbox_to_anchor=(1.04, 1))
        plt.tight_layout()
        return figure

class ImagebasedClassifier(pl.LightningModule):

    def __init__(self, 
            n_classes: int,
            label_def: str,
            loss_name: str,
            learning_rate: float,
            optimizer_name: str,
            momentum: float,
            ignore_index: int,
            mdmc_average: str,
    ):
        super(ImagebasedClassifier, self).__init__()
        self.save_hyperparameters()

        self.num_labels = n_classes
       
        # optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.optimizer_name = optimizer_name

        # loss
        self.ignore_index = ignore_index
        if loss_name == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        # class definitions
        self.label_names, self.label_colors = self._load_label_def(label_def)

        # metrics
        self.mdmc_average = mdmc_average
        self.accuracy_train = torchmetrics.Accuracy(ignore_index=self.ignore_index, 
                mdmc_average=self.mdmc_average)
        self.accuracy_val = torchmetrics.Accuracy(ignore_index=self.ignore_index,
                mdmc_average=self.mdmc_average)
        self.accuracy_test = torchmetrics.Accuracy(ignore_index=self.ignore_index,
                mdmc_average=self.mdmc_average)
        self.f1_train = torchmetrics.F1(ignore_index=self.ignore_index,
                mdmc_average=self.mdmc_average)
        self.f1_val = torchmetrics.F1(ignore_index=self.ignore_index, 
                mdmc_average=self.mdmc_average)
        self.f1_test = torchmetrics.F1(ignore_index=self.ignore_index,
                mdmc_average=self.mdmc_average)
        self.iou_train = torchmetrics.IoU(ignore_index=self.ignore_index, 
                num_classes=self.num_labels+1) # number of labels + undefined
        self.iou_val = torchmetrics.IoU(ignore_index=self.ignore_index, 
                num_classes=self.num_labels+1) # number of labels + undefined
        self.iou_test = torchmetrics.IoU(ignore_index=self.ignore_index, 
                num_classes=self.num_labels+1) # number of labels + undefined
        self.confmat_train = torchmetrics.ConfusionMatrix(
                num_classes=self.num_labels+1) # number of labels + undefined
        self.confmat_val=torchmetrics.ConfusionMatrix(
                num_classes=self.num_labels+1) # number of labels + undefined
        self.confmat_test = torchmetrics.ConfusionMatrix(
                num_classes=self.num_labels+1) # number of labels + undefined

    # pretrain_routine hook
    def on_pretrain_routine_start(self):
        # logging
        self.confmat_log_dir = Path(self.logger.log_dir).joinpath('confmats')
        self.confmat_log_dir.mkdir(parents=True, exist_ok=True)
    
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

        prediction = prediction.argmax(dim=1, keepdims=True)
        labels = labels.unsqueeze(1)
        self.accuracy_train(prediction, labels)
        self.f1_train(prediction, labels)
        self.iou_train(prediction, labels)
        self.confmat_train(prediction, labels)
        self.log('train_loss_step', loss)
        return loss

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.accuracy_train.compute())
        self.log('train_f1_epoch', self.f1_train.compute())
        self.log('train_iou_epoch', self.iou_train.compute())
        confmat_epoch = self.confmat_train.compute().detach().cpu().numpy().astype(np.int)
        confmat_logpath = self.confmat_log_dir.joinpath(f'confmat_train_epoch{self.current_epoch}.csv')
        self._export_confmat(confmat_logpath, confmat_epoch)
        self.logger.experiment.add_figure("Confusion Matrix (train)", 
                self._plot_confmat(confmat_epoch),
                self.current_epoch)

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        prediction = self.forward(inputs)
        loss = self.criterion(prediction, labels.squeeze(dim=1))

        prediction = prediction.argmax(dim=1, keepdims=True)
        labels = labels.unsqueeze(1)
        self.accuracy_val(prediction, labels)
        self.f1_val(prediction, labels)
        self.iou_val(prediction, labels)
        self.confmat_val(prediction, labels)
        self.log('val_loss_step', loss)
        if batch_idx == 0:
            self.logger.experiment.add_figure("Prediction (sample batch):",
                    self._plot_batch_prediction(prediction.detach().cpu().numpy().astype(np.int),
                        labels.detach().cpu().numpy().astype(np.int)),
                    self.current_epoch)


    def validation_epoch_end(self, outs):
        self.log('val_acc_epoch', self.accuracy_val.compute())
        self.log('val_f1_epoch', self.f1_val.compute())
        self.log('val_iou_epoch', self.iou_val.compute())
        confmat_epoch = self.confmat_val.compute().detach().cpu().numpy().astype(np.int)
        confmat_logpath = self.confmat_log_dir.joinpath(f'confmat_val_epoch{self.current_epoch}.csv')
        self._export_confmat(confmat_logpath, confmat_epoch)
        self.logger.experiment.add_figure("Confusion Matrix (validation)", 
                self._plot_confmat(confmat_epoch),
                self.current_epoch)

    def test_step(self, test_batch, batch_idx):
        inputs, labels = test_batch
        prediction = self.forward(inputs)
        loss = self.criterion(prediction, labels.squeeze(dim=1))

        prediction = prediction.argmax(dim=1, keepdims=True)
        labels = labels.unsqueeze(1)
        self.accuracy_test(prediction, labels)
        self.f1_test(prediction, labels)
        self.iou_test(prediction, labels)
        self.confmat_test(prediction, labels)
        self.log('test_loss_step', loss)

    def test_epoch_end(self, outs):
        self.log('test_acc_epoch', self.accuracy_test.compute())
        self.log('test_f1_epoch', self.accuracy_test.compute())
        self.log('test_iou_epoch', self.accuracy_test.compute())
        confmat_epoch = self.confmat_test.compute().detach().cpu().numpy().astype(np.int)
        confmat_logpath = self.confmat_log_dir.joinpath(f'confmat_test_epoch{self.current_epoch}.csv')
        self._export_confmat(confmat_logpath, confmat_epoch)
        self.logger.experiment.add_figure("Confusion Matrix (test)", 
                self._plot_confmat(confmat_epoch),
                self.current_epoch)

    def _export_confmat(self, path, confmat):
        np.savetxt(path, confmat)

    def _plot_confmat(self, confmat):
        figure = plt.figure()
        plt.matshow(confmat[:-1,:-1], fignum=0)
        return figure

    def _load_label_def(self, label_def):
        label_defs = np.loadtxt(label_def, delimiter=',', dtype=str)
        label_names = np.array(label_defs[:,1])
        label_colors = np.array(label_defs[:, 2:], dtype='int')
        return label_names, label_colors

    def _plot_batch_prediction(self, pred, label):
        batch_size = label.shape[0]
        figure, axes = plt.subplots(nrows=2, ncols=batch_size, squeeze=False)
        for i in range(batch_size):
            axes[0, i].imshow(self.label_colors[label[i]].squeeze())
            axes[1, i].imshow(self.label_colors[pred[i]].squeeze())

        # add legend that shows label color class mapping
        handles = []
        for i, color in enumerate(self.label_colors):
            handles.append(mpatches.Patch(color=color*(1./255), label=self.label_names[i]))

        plt.legend(handles=handles, loc='center left', ncol=2, bbox_to_anchor=(1.04, 1))
        plt.tight_layout()
        return figure
