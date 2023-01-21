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

from typing import Optional

def inv_num_of_samples(histogram):
    return histogram.sum()/histogram

def inv_square_num_of_samples(histogram):
    return histogram.sum()/torch.sqrt(histogram)

class SemanticSegmentationModule(pl.LightningModule):
    
    def __init__(self, 
            n_channels: int,
            n_classes: int,
            label_def: str,
            loss_name: str,
            learning_rate: float,
            optimizer_name: str,
            momentum: float,
            weight_decay: float,
            ignore_index: int,
            mdmc_average: str,
            classification_task: str = "multiclass",
            class_weighting: str = None,
            **kwargs
    ):
        super(SemanticSegmentationModule, self).__init__(**kwargs)

        self.hparams["model_name"] = type(self).__name__

        self.n_classes = n_classes
        self.n_channels = n_channels
       
        # optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.optimizer_name = optimizer_name

        # loss
        self.ignore_index = ignore_index
        self.class_weighting = class_weighting
        self.loss_name = 'cross_entropy'

        # class definitions
        self.label_names, self.label_colors = self._load_label_def(label_def)

        # metrics
        self.export_metrics = True
        self.mdmc_average = mdmc_average
        self.classification_task = classification_task
        

        ## Attention! Unfortunately, metrics must be created as members instead of directly storing
        ## them in dictionaries otherwise they are not identified as child modules
        ## and in turn not moved to the correct device
        ## TODO This is only a workaround, I should find a better solution in the future

        ## train metrics
        self.train_metrics = {}
        self.acc_train_micro = torchmetrics.Accuracy(
                task=self.classification_task,
                num_classes=self.n_classes, 
                ignore_index=self.ignore_index, 
                mdmc_average=self.mdmc_average, 
                average='micro')
        self.train_metrics["Train/accuracy-micro"] = self.acc_train_micro
        self.acc_train_macro = torchmetrics.Accuracy(
                task=self.classification_task,
                num_classes=self.n_classes,
                ignore_index=self.ignore_index, 
                mdmc_average=self.mdmc_average, 
                average='macro')
        self.train_metrics["Train/accuracy-macro"] = self.acc_train_macro
        self.acc_train_class = torchmetrics.Accuracy(
                task=self.classification_task,
                num_classes=self.n_classes,
                ignore_index=self.ignore_index, 
                mdmc_average=self.mdmc_average, 
                average='none')
        self.train_metrics["Train/accuracy-class"] = self.acc_train_class
        self.f1_train_micro = torchmetrics.F1Score(
                task=self.classification_task,
                num_classes=self.n_classes,
                ignore_index=self.ignore_index, 
                mdmc_average=self.mdmc_average, 
                average='micro')
        self.train_metrics["Train/f1-micro"] = self.f1_train_micro
        self.f1_train_macro = torchmetrics.F1Score(
                task=self.classification_task,
                num_classes=self.n_classes,
                ignore_index=self.ignore_index, 
                mdmc_average=self.mdmc_average, 
                average='macro')
        self.train_metrics["Train/f1-macro"] = self.f1_train_macro
        self.f1_train_class = torchmetrics.F1Score(
                task=self.classification_task,
                 num_classes=self.n_classes,
                ignore_index=self.ignore_index, 
                mdmc_average=self.mdmc_average, 
                average='none')
        self.train_metrics["Train/f1-class"] = self.f1_train_class
        self.jaccard_train = torchmetrics.JaccardIndex(
                task=self.classification_task,
                ignore_index=self.ignore_index, 
                num_classes=self.n_classes)
        self.train_metrics["Train/jaccard"] = self.jaccard_train
        self.jaccard_train_class = torchmetrics.JaccardIndex(
                task=self.classification_task,
                average='none',
                ignore_index=self.ignore_index,
                num_classes=self.n_classes)
        self.train_metrics["Train/jaccard-class"] = self.jaccard_train_class
        self.confmat_train = torchmetrics.ConfusionMatrix(
                task=self.classification_task,
                num_classes=self.n_classes,
                ignore_index=self.ignore_index)
        self.train_metrics["Train/conf_mat"] = self.confmat_train

        ## val metrics
        self.val_metrics = {}
        self.acc_val_micro = torchmetrics.Accuracy(
                task=self.classification_task,
                num_classes=self.n_classes,
                ignore_index=self.ignore_index, 
                mdmc_average=self.mdmc_average, 
                average='micro')
        self.val_metrics["Validation/accuracy-micro"] = self.acc_val_micro
        self.acc_val_macro = torchmetrics.Accuracy(
                task=self.classification_task,
                num_classes=self.n_classes,
                ignore_index=self.ignore_index, 
                mdmc_average=self.mdmc_average, 
                average='macro')
        self.val_metrics["Validation/accuracy-macro"] = self.acc_val_macro
        self.acc_val_class = torchmetrics.Accuracy(
                task=self.classification_task,
                num_classes=self.n_classes,
                ignore_index=self.ignore_index, 
                mdmc_average=self.mdmc_average, 
                average='none')
        self.val_metrics["Validation/accuracy-class"] = self.acc_val_class
        self.f1_val_micro = torchmetrics.F1Score(
                task=self.classification_task,
                num_classes=self.n_classes,
                ignore_index=self.ignore_index, 
                mdmc_average=self.mdmc_average, 
                average='micro')
        self.val_metrics["Validation/f1-micro"] = self.f1_val_micro
        self.f1_val_macro = torchmetrics.F1Score(
                task=self.classification_task,
                num_classes=self.n_classes,
                ignore_index=self.ignore_index, 
                mdmc_average=self.mdmc_average, 
                average='macro')
        self.val_metrics["Validation/f1-macro"] = self.f1_val_macro
        self.f1_val_class = torchmetrics.F1Score(
                task=self.classification_task,
                num_classes=self.n_classes,
                ignore_index=self.ignore_index, 
                mdmc_average=self.mdmc_average, 
                average='none')
        self.val_metrics["Validation/f1-class"] = self.f1_val_class
        self.jaccard_val = torchmetrics.JaccardIndex(
                task=self.classification_task,
                ignore_index=self.ignore_index, 
                num_classes=self.n_classes)
        self.val_metrics["Validation/jaccard"] = self.jaccard_val
        self.jaccard_val_class = torchmetrics.JaccardIndex(
                task=self.classification_task,
                average='none',
                ignore_index=self.ignore_index,
                num_classes=self.n_classes)
        self.val_metrics["Validation/jaccard-class"] = self.jaccard_val_class
        self.confmat_val = torchmetrics.ConfusionMatrix(
                task=self.classification_task,
                num_classes=self.n_classes,
                ignore_index=self.ignore_index)
        self.val_metrics["Validation/conf_mat"] = self.confmat_val

        ## test metrics
        self.test_metrics = {}
        self.acc_test_micro = torchmetrics.Accuracy(
                task=self.classification_task,
                num_classes=self.n_classes,
                ignore_index=self.ignore_index, 
                mdmc_average=self.mdmc_average, 
                average='micro')
        self.test_metrics["Test/accuracy-micro"] = self.acc_test_micro
        self.acc_test_macro = torchmetrics.Accuracy(
                task=self.classification_task,
                num_classes=self.n_classes,
                ignore_index=self.ignore_index, 
                mdmc_average=self.mdmc_average, 
                average='macro')
        self.test_metrics["Test/accuracy-macro"] = self.acc_test_macro
        self.acc_test_class = torchmetrics.Accuracy(
                task=self.classification_task,
                num_classes=self.n_classes,
                ignore_index=self.ignore_index, 
                mdmc_average=self.mdmc_average, 
                average='none')
        self.test_metrics["Test/accuracy-class"] = self.acc_test_class
        self.f1_test_micro = torchmetrics.F1Score(
                task=self.classification_task,
                num_classes=self.n_classes,
                ignore_index=self.ignore_index, 
                mdmc_average=self.mdmc_average, 
                average='micro')
        self.test_metrics["Test/f1-micro"] = self.f1_test_micro
        self.f1_test_macro = torchmetrics.F1Score(
                task=self.classification_task,
                num_classes=self.n_classes,
                ignore_index=self.ignore_index, 
                mdmc_average=self.mdmc_average, 
                average='macro')
        self.test_metrics["Test/f1-macro"] = self.f1_test_macro
        self.f1_test_class = torchmetrics.F1Score(
                task=self.classification_task,
                num_classes=self.n_classes,
                ignore_index=self.ignore_index, 
                mdmc_average=self.mdmc_average, 
                average='none')
        self.test_metrics["Test/f1-class"] = self.f1_test_class
        self.jaccard_test = torchmetrics.JaccardIndex(
                task=self.classification_task,
                ignore_index=self.ignore_index, 
                num_classes=self.n_classes)
        self.test_metrics["Test/jaccard"] = self.jaccard_test
        self.jaccard_test_class = torchmetrics.JaccardIndex(
                task=self.classification_task,
                average='none',
                ignore_index=self.ignore_index,
                num_classes=self.n_classes)
        self.test_metrics["Test/jaccard-class"] = self.jaccard_test_class
        self.confmat_test = torchmetrics.ConfusionMatrix(
                task=self.classification_task,
                num_classes=self.n_classes,
                ignore_index=self.ignore_index)
        self.test_metrics["Test/conf_mat"] = self.confmat_test

    def setup(self, stage: Optional[str]=None):
        if self.class_weighting is not None:
            train_c_hist, _, _ = self.trainer.datamodule.class_histograms() 
            if self.class_weighting == 'INS':
                self.class_weights = inv_num_of_samples(train_c_hist)
            elif self.class_weighting == 'ISNS':
                self.class_weights = inv_square_num_of_samples(train_c_hist)
            else :
                raise RuntimeError(f'Class weighting strategy "{self.class_weighting}" does'
                        ' not exist or is not implemented yet!')
        else:
            self.class_weights = torch.ones(self.n_classes)
        
        if self.loss_name == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss(
                    ignore_index=self.ignore_index,
                    weight=self.class_weights)
        else : 
            raise RuntimeError(f'Loss function "{self.loss_name}" is not available '
                    'or is not implemented yet')

        # logging
        if self.export_metrics:
            # confusion matrices
            self.confmat_log_dir = Path(self.logger.log_dir).joinpath('confmats')
            self.confmat_log_dir.mkdir(parents=True, exist_ok=True)

            # class-wise metrics
            self.classmetric_log_dir = Path(self.logger.log_dir).joinpath('class-metrics')
            self.classmetric_log_dir.mkdir(parents=True, exist_ok=True)

    def configure_optimizers(self):
        if self.optimizer_name == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), 
                    lr=self.learning_rate, 
                    momentum=self.momentum,
                    weight_decay=self.weight_decay)
        elif self.optimizer_name == 'RMSprop':
            optimizer = torch.optim.RMSprop(self.parameters(),
                    lr=self.learning_rate,
                    momentum=self.momentum,
                    weight_decay=self.weight_decay)
        elif self.optimizer_name == 'Adam':
            optimizer = torch.optim.Adam(self.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay)
        elif self.optimizer_name == 'AdamW':
            # TODO add variable to adjust epsilon for AdamW
            optimizer = torch.optim.AdamW(self.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay)
        else :
            raise RuntimeError(f'Optimizer {self.optimizer_name} unknown!')

        return optimizer

    def training_step(self, train_batch, batch_idx):
        inputs, labels = train_batch
        prediction = self.forward(inputs)
        '''
        print("===== IMAGES ====")
        print(inputs.dtype)
        print(inputs.element_size())
        print(inputs.nelement())
        print(inputs.element_size()*inputs.nelement())
        print("===== LABELS ====")        
        print(labels.dtype)
        print(labels.element_size())
        print(labels.nelement())
        print(labels.element_size()*labels.nelement())
        '''

        # convert prediction to same shape as ground-truth labels
        # (? pretty sure this is what happens, but I should have
        #  added a comment in the first place)
        prediction =  T.Resize((labels.shape[1:3]))(prediction)
        '''
        print("===== prediction ====")        
        print(prediction.dtype)
        print(prediction.element_size())
        print(prediction.nelement())
        print(prediction.element_size()*prediction.nelement())
        '''

        loss = self.criterion(prediction, labels.squeeze(dim=1)) 
        '''
        print("===== loss ====")        
        print(loss.dtype)
        print(loss.element_size())
        print(loss.nelement())
        print(loss.element_size()*loss.nelement())
        '''

        if self.export_metrics:
            prediction = prediction.argmax(dim=1, keepdims=True)
            labels = labels.unsqueeze(1)
            for name, metric in self.train_metrics.items():
                metric(prediction, labels)

        self.log('train_loss_step', loss)
        return loss

    def training_epoch_end(self, outs):
        if self.export_metrics:
            for name, metric in self.train_metrics.items():
                if "conf_mat" in name:
                    confmat_epoch = metric.compute()

                    # plot confusion_matrix
                    confmat_epoch = confmat_epoch.detach().cpu().numpy().astype(np.int)
                    confmat_logpath = self.confmat_log_dir.joinpath(
                            f'confmat_train_epoch{self.current_epoch}.csv')
                    self._export_confmat(confmat_logpath, confmat_epoch)
                    self.logger.experiment.add_figure("Train/confusion-matrix", 
                            self._plot_confmat(confmat_epoch),
                            self.current_epoch)
                elif "class" in name:
                    # actually compute score from logs
                    score_epoch = metric.compute()
                    
                    # detach from graph, retrieve from gpu-memory, convert to numpy array
                    score_epoch = score_epoch.detach().cpu().numpy()

                    # extract metric name (remove 'Validation/','Train/','Test/' and replace spaces)
                    metric_name = name.split('/')[1].replace(' ', '_')

                    # export and plot classwise scores
                    score_logpath = self.classmetric_log_dir.joinpath(
                            f'{metric_name}_train_epoch{self.current_epoch}.csv')
                    self._export_classmetric(score_logpath, score_epoch)
                    self.logger.experiment.add_figure(f"Train/{metric_name}",
                            self._plot_classmetric(score_epoch),
                            self.current_epoch)
                else:
                    self.log(name, metric.compute())
                    metric.reset()

    def validation_step(self, val_batch, batch_idx):
        inputs, labels = val_batch
        prediction = self.forward(inputs)

        # convert prediction to same shape as ground-truth labels
        # (? pretty sure this is what happens, but I should have
        #  added a comment in the first place)
        # TODO (RB) this doesn't seem to do anything, prediction.shape is the same before
        # and after
        prediction =  T.Resize((labels.shape[1:3]))(prediction)
        loss = self.criterion(prediction, labels.squeeze(dim=1))

        if self.export_metrics:
            prediction = prediction.argmax(dim=1, keepdims=True)
            labels = labels.unsqueeze(1)
            
            for name, metric in self.val_metrics.items():
                metric(prediction, labels)
            
            # Visualize prediction of first batch in each epoch
            if batch_idx == 0:
                self.logger.experiment.add_figure("Prediction (sample batch):",
                        self._plot_batch_prediction(prediction.detach().cpu().numpy().astype(np.int),
                            labels.detach().cpu().numpy().astype(np.int)),
                        self.current_epoch)


    def validation_epoch_end(self, outs):
        if self.export_metrics:
            for name, metric in self.val_metrics.items():
                if "conf_mat" in name:
                    confmat_epoch = metric.compute()

                    # plot confusion_matrix
                    confmat_epoch = confmat_epoch.detach().cpu().numpy().astype(np.int)
                    confmat_logpath = self.confmat_log_dir.joinpath(
                            f'confmat_val_epoch{self.current_epoch}.csv')
                    self._export_confmat(confmat_logpath, confmat_epoch)
                    self.logger.experiment.add_figure("Validation/confusion-matrix", 
                            self._plot_confmat(confmat_epoch),
                            self.current_epoch)
                elif "class" in name:
                    # actually compute score from logs
                    score_epoch = metric.compute()
                    
                    # detach from graph, retrieve from gpu-memory, convert to numpy array
                    score_epoch = score_epoch.detach().cpu().numpy()

                    # extract metric name (remove 'Validation/','Train/','Test/' and replace spaces)
                    metric_name = name.split('/')[1].replace(' ', '_')

                    # export and plot classwise scores
                    score_logpath = self.classmetric_log_dir.joinpath(
                            f'{metric_name}_val_epoch{self.current_epoch}.csv')
                    self._export_classmetric(score_logpath, score_epoch)
                    self.logger.experiment.add_figure(f"Validation/{metric_name}",
                            self._plot_classmetric(score_epoch),
                            self.current_epoch)
                else :
                    self.log(name, metric.compute())
                    metric.reset()

    def test_step(self, test_batch, batch_idx):
        inputs, labels = test_batch
        prediction = self.forward(inputs)

        # convert prediction to same shape as ground-truth labels
        # (? pretty sure this is what happens, but I should have
        #  added a comment in the first place)
        prediction =  T.Resize((labels.shape[1:3]))(prediction)
        loss = self.criterion(prediction, labels.squeeze(dim=1))
        
        if self.export_metrics:
            prediction = prediction.argmax(dim=1, keepdims=True)
            labels = labels.unsqueeze(1)

            for _, metric in self.test_metrics.items():
                metric(prediction, labels)

    def test_epoch_end(self, outs):
        if self.export_metrics:
            for name, metric in self.test_metrics.items():
                if "conf_mat" in name:
                    confmat_epoch = metric.compute()

                    # plot confusion_matrix
                    confmat_epoch = confmat_epoch.detach().cpu().numpy().astype(np.int)
                    confmat_logpath = self.confmat_log_dir.joinpath(
                            f'confmat_val_epoch{self.current_epoch}.csv')
                    self._export_confmat(confmat_logpath, confmat_epoch)
                    self.logger.experiment.add_figure("Test/confusion-matrix", 
                            self._plot_confmat(confmat_epoch),
                            self.current_epoch)
                elif "class" in name:
                    # actually compute score from logs
                    score_epoch = metric.compute()
                    
                    # detach from graph, retrieve from gpu-memory, convert to numpy array
                    score_epoch = score_epoch.detach().cpu().numpy()

                    # extract metric name (remove 'Validation/','Train/','Test/' and replace spaces)
                    metric_name = name.split('/')[1].replace(' ', '_')

                    # export and plot classwise scores
                    score_logpath = self.classmetric_log_dir.joinpath(
                            f'{metric_name}_test_epoch{self.current_epoch}.csv')
                    self._export_classmetric(score_logpath, score_epoch)
                    self.logger.experiment.add_figure(f"Test/{metric_name}",
                            self._plot_classmetric(score_epoch),
                            self.current_epoch)

                else :
                    self.log(name, metric.compute())
                    metric.reset()

    def _export_confmat(self, path, confmat):
        np.savetxt(path, confmat)

    def _plot_confmat(self, confmat):
        fig, ax = plt.subplots()
        label_ids = np.arange(confmat.shape[0]-1)
        ax.matshow(confmat[:-1,:-1])
        ax.set_xticks(label_ids)
        ax.set_xticklabels([self.label_names[i] for i in label_ids], rotation=90)
        ax.set_yticks(label_ids)
        ax.set_yticklabels([self.label_names[i] for i in label_ids])
        plt.tight_layout()

        return fig

    def _export_classmetric(self, path, scores):
        np.savetxt(path, scores)

    def _plot_classmetric(self, scores):
        fig, ax = plt.subplots()
        label_ids = np.arange(scores.shape[0])
        ax.bar(label_ids, scores)
        ax.set_xticks(label_ids)
        ax.set_xticklabels([self.label_names[i] for i in label_ids], rotation=90)
        plt.tight_layout()
        return fig

    def _load_label_def(self, label_def):
        label_defs = np.loadtxt(label_def, delimiter=',', dtype=str)
        label_names = np.array(label_defs[:,1])
        label_colors = np.array(label_defs[:, 2:], dtype='int')
        return label_names, label_colors

    def _plot_batch_prediction(self, pred, label):
        batch_size = label.shape[0]
        # four predictions per row, followed by four labelimages
        n_cols = 4
        n_rows = 2 * int((batch_size + 3)/4)
        figure, axes = plt.subplots(nrows=n_rows, ncols=n_cols, squeeze=False, figsize=(12,12))
        for i in range(batch_size):
            r = i % 4
            c = 2 * int(i / 4)
            
            axes[c, r].imshow(self.label_colors[label[i]].squeeze())
            axes[c+1, r].imshow(self.label_colors[pred[i]].squeeze())

            # remove axes for cleaner image
            axes[c, r].axis('off')
            axes[c+1, r].axis('off')

        # add legend that shows label color class mapping
        handles = []
        for i, color in enumerate(self.label_colors):
            handles.append(mpatches.Patch(color=color*(1./255), label=self.label_names[i]))

        figure.legend(handles=handles, loc='lower left', ncol=4, mode='expand')
        figure.tight_layout()
        return figure
