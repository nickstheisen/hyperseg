#!/usr/bin/env python
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from torchvision import transforms

import h5py

from typing import List, Any, Optional
from pathlib import Path

import numpy as np
from hyperseg.datasets.analysis.tools import StatCalculator
from hyperseg.datasets.transforms import Normalize

def label_histogram(dataset, n_classes):
    label_hist = torch.zeros(n_classes) # do not count 'unefined'(highest class_id)
    for i, (_, labels) in enumerate(DataLoader(dataset)):
        label_ids, counts = labels.unique(return_counts=True)
        for i in range(len(label_ids)):
            label_id = label_ids[i]
            if not (label_id == n_classes):
                label_hist[label_id] += counts[i]
    return label_hist

class HSDataModule(pl.LightningDataModule):

    def __init__(
            self,
            filepath: str,
            num_workers: int,
            batch_size: int,
            train_prop: float, # train proportion (of all data)
            val_prop: float, # validation proportion (of all data)
            label_def: str,
            manual_seed: int=None,
            precalc_histograms: bool=False,
            normalize: bool=False,
            spectral_average: bool=False,
            prep_3dconv: bool=False,
    ):
        super().__init__()
        
        self.save_hyperparameters()

        self.filepath = filepath
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.train_prop = train_prop
        self.val_prop = val_prop
        self.manual_seed = manual_seed
        
        self.label_def = label_def

        self.spectral_average = spectral_average
        self.prep_3dconv = prep_3dconv
        
        self.precalc_histograms=precalc_histograms
        self.c_hist_train = None
        self.c_hist_val = None
        self.c_hist_test = None

        # statistics (if normalization is activated)
        self.normalize = normalize
        self.means = None
        self.stds = None


    def class_histograms(self):
        if self.c_hist_train is not None :
            return (self.c_hist_train, self.c_hist_val, self.c_hist_test)
        else :
            return None
    
    def effective_sample_counts(self):
        if self.n_train_samples is not None :
            return (self.n_train_samples, self.n_val_samples, self.n_test_samples)
        else :
            return None

    def setup(self, stage: Optional[str] = None):
        dataset = GroundBasedHSDataset(self.filepath, transform=self.transform)
        train_size = round(self.train_prop * len(dataset))
        val_size = round(self.val_prop * len(dataset))
        test_size = len(dataset) - (train_size + val_size)
        if self.manual_seed is not None:
            self.dataset_train, self.dataset_val, self.dataset_test = random_split(
                    dataset, 
                    [train_size, val_size, test_size], 
                    generator=torch.Generator().manual_seed(self.manual_seed))
        else :
            self.dataset_train, self.dataset_val, self.dataset_test = random_split(
                    dataset, 
                    [train_size, val_size, test_size])

        # calculate class_histograms
        if self.precalc_histograms:
            self.c_hist_train = label_histogram(
                    self.dataset_train, self.n_classes)
            self.c_hist_val = label_histogram(
                    self.dataset_val, self.n_classes)
            self.c_hist_test = label_histogram(
                    self.dataset_test, self.n_classes)

        # calculate data statistics for normalization
        if self.normalize:
            stat_calc = StatCalculator(self.dataset_train)
            self.means, self.stds = stat_calc.getDatasetStats()

            # enable normalization in whole data set
            dataset.enable_normalization(self.means, self.stds)
                
    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)

class GroundBasedHSDataset(Dataset):

    def __init__(self, filepath, transform):
        self._filepath = filepath
        
        # if h5file is kept open, the object cannot be pickled and in turn 
        # multi-gpu cannot be used
        h5file = h5py.File(self._filepath, 'r')
        self._samplelist = list(h5file.keys())
        self._transform = transform
        h5file.close()

    def __len__(self):
        return len(self._samplelist)

    def samplelist(self):
        return self._samplelist
    
    def enable_normalization(self, means, stds):
        self._transform = transforms.Compose([
            self._transform,
            Normalize(means=means, stds=stds)
        ])
        self.mean = means
        self.std = stds

    def __getitem__(self, idx):
        h5file = h5py.File(self._filepath)
        sample = (np.array(h5file[self._samplelist[idx]]['data']),
                np.array(h5file[self._samplelist[idx]]['labels']))

        if self._transform:
            sample = self._transform(sample)
        
        h5file.close()
        return sample
