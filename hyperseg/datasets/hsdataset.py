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
from hyperseg.datasets.utils import label_histogram

N_DEBUG_SAMPLES = 5

class HSDataModule(pl.LightningDataModule):

    def __init__(
            self,
            basepath: str,
            num_workers: int,
            batch_size: int,
            label_def: str,
            manual_seed: int=None,
            normalize: bool=False,
            spectral_average: bool=False,
            pca: int=None,
            pca_out_dir: str='.',
            prep_3dconv: bool=False,
            debug: bool = False
    ):
        super().__init__()
        
        self.save_hyperparameters()

        self.basepath = Path(basepath)
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.manual_seed = manual_seed
        self.debug = debug
        
        self.label_def = label_def

        
        # data preprocessing
        self.normalize = normalize
        self.spectral_average = spectral_average
        self.pca=pca
        self.pca_out_dir=Path(pca_out_dir)
        if not self.pca_out_dir.exists():
            self.pca_out_dir.mkdir(parents=True, exist_ok=True)
        if not self.pca_out_dir.is_dir():
            raise RuntimeError("`pca_out_dir` must be a directory!")
        self.prep_3dconv = prep_3dconv
      
        '''
        self.precalc_histograms=precalc_histograms
        self.c_hist_train = None
        self.c_hist_val = None
        self.c_hist_test = None

        '''


    '''
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
    '''
    
    
    def setup(self, stage: Optional[str] = None):
        raise NotImplementedError()

    def enable_normalization(self):
        stat_calc = StatCalculator(self.dataset_train)
        self.means, self.stds = stat_calc.getDatasetStats()

        # enable normalization in whole data set
        dataset_train.enable_normalization(self.means, self.stds)
        dataset_test.enable_normalization(self.means, self.stds)
        dataset_val.enable_normalization(self.means, self.stds)

    def enable_pca(self):
        raise NotImplementedError

    def train_dataloader(self):
        return DataLoader(self.dataset_train,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=self.num_workers,
                persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.dataset_val,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.dataset_test,
                batch_size=self.batch_size,
                shuffle=False,
                pin_memory=True,
                num_workers=self.num_workers)

class HSDataset(Dataset):

    def __init__(self, filepath, transform, debug=False):
        self._filepath = filepath
        
        # if h5file is kept open, the object cannot be pickled and in turn 
        # multi-gpu cannot be used
        h5file = h5py.File(self._filepath, 'r')
        self.debug = debug
        self._samplelist = list(h5file.keys())
        self._transform = transform

        if self.debug:
            # only use N_DEBUG_SAMPLES samples overall
            self._samplelist = self._samplelist[:N_DEBUG_SAMPLES]
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
        sample = (np.array(h5file[self._samplelist[idx]]['image']),
                np.array(h5file[self._samplelist[idx]]['labels']))

        if self._transform:
            sample = self._transform(sample)
        
        h5file.close()
        return sample
