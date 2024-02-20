#!/usr/bin/env python

import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from torchvision import transforms


from hyperseg.datasets.analysis.tools import StatCalculator
from hyperseg.datasets.transforms import ToTensor, ReplaceLabels, Normalize, SpectralAverage, InsertEmptyChannelDim, PermuteData
from .hsdataset import HSDataModule, HSDataset
from hyperseg.datasets.utils import apply_pca

from typing import List, Any, Optional
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from imageio import imread

def label_histogram(dataset, n_classes):
    label_hist = torch.zeros(n_classes) # do not count 'unefined'(highest class_id)
    for i, (_, labels) in enumerate(DataLoader(dataset)):
        label_ids, counts = labels.unique(return_counts=True)
        for i in range(len(label_ids)):
            label_id = label_ids[i]
            if not (label_id == n_classes):
                label_hist[label_id] += counts[i]
    return label_hist

class HSIDriveDataset(Dataset):

    def __init__(self, basepath, transform, debug=False):
        self._basepath = Path(basepath)
        self._labelpath = self._basepath.joinpath('labels')
        self._datapath = self._basepath.joinpath('cubes_fl32')
        self.debug = debug
        self._samplelist = np.array(
            [sample.stem for sample in self._labelpath.glob('*.png')]
        )
        
        if self.debug:
            self._samplelist = self._samplelist[:100]
        
        self._transform = transform

    def __len__(self):
        return len(self._samplelist)

    def samplelist(self):
        return self._samplelist

    def enable_normalization(self, means, stds):
        self._transform = transforms.Compose([
            self._transform,
            Normalize(means=means, stds=stds)
        ])

    def __getitem__(self, idx):
        samplename = self._samplelist[idx]
        data = loadmat(self._datapath.joinpath(samplename + "_TC.mat"))['cube']
        label = imread(self._labelpath.joinpath(samplename + ".png"))

        sample = (data, label)

        if self._transform:
            sample = self._transform(sample)
        return sample

class HSIDrive(HSDataModule):
    def __init__(self, ignore_water:bool=True, **kwargs):
        super().__init__(**kwargs)
        
        self.save_hyperparameters()

        self.ignore_water = ignore_water
        self.n_classes = 10 if self.ignore_water else 11
        self.undef_idx = 9 if self.ignore_water else 10

        self.filepath_train = self.basepath.joinpath('hsidrive_train.h5')
        self.filepath_test = self.basepath.joinpath('hsidrive_test.h5')
        self.filepath_val = self.basepath.joinpath('hsidrive_val.h5')

        if ignore_water:
            self.transform = transforms.Compose([
                            ToTensor(),
                            ReplaceLabels({0:9, 1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:9, 9:7, 10:8}), # replace undefined 0 and water labels 8 with 9 and then shift labels according
                            PermuteData(new_order=[2,0,1]),
                        ])
        else:
            self.transform = transforms.Compose([
                            ToTensor(),
                            ReplaceLabels({0:10, 1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9}), # replace undefined label 0 with 10 and then shift labels by one
                            PermuteData(new_order=[2,0,1]),
                        ])
        if self.spectral_average:
            self.transform = transforms.Compose([
                                self.transform,
                                SpectralAverage()
                             ])

        if self.pca is not None:
            self.enable_pca()
        
        # read dimensions from image
        dataset = HSDataset(self.filepath_val, transform=self.transform)
        img, _ = dataset[1]
        self.img_shape = img.shape[1:]
        self.n_channels = img.shape[0]
       
    def setup(self, stage: Optional[str] = None):       
        self.dataset_train = HSDataset(  self.filepath_train, 
                                    transform=self.transform,
                                    debug=self.debug)
        self.dataset_test = HSDataset(   self.filepath_test,
                                    transform=self.transform,
                                    debug=self.debug)
        self.dataset_val = HSDataset(    self.filepath_val,
                                    transform=self.transform,
                                    debug=self.debug)
        
        # calculate data statistics for normalization
        if self.normalize:
            self.enable_normalization()

    def enable_pca(self):
        # train
        outpath_train = self.pca_out_dir.joinpath(f'hsidrive_train_pca{self.pca}.h5')
        apply_pca(  self.pca, self.filepath_train, outpath_train, 
                    debug=self.debug, half_precision=False)
        self.filepath_train = outpath_train

        # test
        outpath_test = self.pca_out_dir.joinpath(f'hsidrive_test_pca{self.pca}.h5')
        apply_pca(  self.pca, self.filepath_test, outpath_test, 
                    debug=self.debug, half_precision=False)
        self.filepath_test = outpath_test

        # val 
        outpath_val = self.pca_out_dir.joinpath(f'hsidrive_val_pca{self.pca}.h5')
        apply_pca(  self.pca, self.filepath_val, outpath_val, 
                    debug=self.debug, half_precision=False)
        self.filepath_val = outpath_val
