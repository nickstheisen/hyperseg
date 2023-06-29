#!/usr/bin/env python

from pathlib import Path
from scipy.io import loadmat, savemat
import numpy as np
from sklearn.decomposition import PCA
import warnings
from math import ceil
from typing import List, Any
import h5py

import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision import transforms

from .prep import download_dataset, split_random_sampling, split_valid_sampling
from hyperseg.datasets.transforms import ToTensor, InsertEmptyChannelDim, PermuteData


class HSDataModule(pl.LightningDataModule):

    def __init__(
            self,
            base_dir: str,
            num_workers: int = 8,
            batch_size: int = 32,
            dataset_name: str = 'AeroRIT_radiance_mid',
            sampling_method: str = 'valid',
            patch_size: int = 7,
            train_prop: float = 0.7, # train proportion (of all data)
            val_prop: float = 0.1, # validation proportion (of all data)
            padding_mode: str = 'constant',
            padding_values = 0, # List[int] or int
            ignore_labels: List[int] = [0],
            apply_pca: bool = False,
            pca_dim: int = 75,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.dataset_name = dataset_name
        self.sampling_method = sampling_method
        self.patch_size = patch_size
        self.train_prop = train_prop
        self.val_prop = val_prop
        self.padding_mode = padding_mode
        self.padding_values = padding_values
        self.ignore_labels = ignore_labels
        self.apply_pca = apply_pca
        self.pca_dim = pca_dim

    def prepare_data(self):
        # download data if not already existing
        self.dataset_path = download_dataset(base_dir=self.base_dir, scene=self.dataset_name)

        self.transform = transforms.Compose([
            ToTensor(),
            InsertEmptyChannelDim(),
            PermuteData(new_order=[0,3,1,2])])

        # sample data and split into train and test set
        if self.sampling_method == 'valid':
            self.filepath = split_valid_sampling(
                    inpath=self.dataset_path, 
                    patch_size=self.patch_size, 
                    train_prop=self.train_prop, 
                    val_prop=self.val_prop,
                    outpath=self.dataset_path.parents[0],
                    padding_mode=self.padding_mode,
                    padding_values=self.padding_values,
                    ignore_labels=self.ignore_labels)
        elif self.sampling_method == 'random':
            self.filepath = split_random_sampling(
                    inpath=self.dataset_path, 
                    patch_size=self.patch_size, 
                    train_prop=self.train_prop, 
                    val_prop=self.val_prop,
                    outpath=self.dataset_path.parents[0],
                    padding_mode=self.padding_mode,
                    padding_values=self.padding_values,
                    ignore_labels=self.ignore_labels)
        else:
            raise RuntimeError(f'Sampling method {self.sampling_method} unknonw')
        
    def train_dataloader(self):
        train_set = RemoteSensingDataset(
                filepath=self.filepath,
                split_type='train',
                apply_pca=self.apply_pca,
                pca_dim=self.pca_dim,
                transform=self.transform)
        train_loader = DataLoader(train_set, 
                batch_size=self.batch_size, 
                shuffle=True, 
                num_workers=self.num_workers)
        return train_loader


    def val_dataloader(self):
        val_set = RemoteSensingDataset(
                filepath=self.filepath,
                split_type='validation',
                apply_pca=self.apply_pca,
                pca_dim=self.pca_dim,
                transform=self.transform)
        val_loader = DataLoader(val_set,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
        return val_loader

    def test_dataloader(self):
        test_set = RemoteSensingDataset(
                filepath=self.filepath,
                split_type='test',
                apply_pca=self.apply_pca,
                pca_dim=self.pca_dim,
                transform=self.transform)
        test_loader = DataLoader(test_set,
                batch_isze=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)
        return test_loader

class RemoteSensingDataset(Dataset):
    """
    Base class to represent hyperspectral remote sensing data sets.

    Attributes
    ----------
    _filepath : PosixPath
        Path to file containing train and test data (.h5, .hdf5).
    _split_type : str
        Defines if train, validation or test data is returned.
    _apply_pca : bool
        Defines if dimensionality is reduced with PCA.
    _pca_dim : int
        Number of dimensions that data is reduced to.
    _transform : functional
        Transformations applied to data samples.
    _pca : object
        Stores sklearn.decomposition.PCA object that is used to reduced dimensionality

    Methods
    -------

    __len__():
        Return length of data set.

    __getitem__(self, idx):
        Return sample in data set at position 'idx'.

    _pca(self, data):
        Applies PCA to reduce datas spectral dimensionality.

    """
    def __init__(self, filepath, split_type,
                 apply_pca=False,
                 pca_dim=75,
                 transform=None):

        self._file = h5py.File(filepath, 'r')
        if split_type == 'train':
            self._samplelist = self._file['trainsample_list']
        elif split_type == 'validation':
            self._samplelist = self._file['valsample_list']
        elif split_type == 'test':
            self._samplelist = self._file['testsample_list']
        else:
            raise RuntimeError(f'`split_type` {split_type} is unknown. Use `train`, `test` '
                    'or `validation instead!')

        self._transform = transform
        self._apply_pca = apply_pca
        self._pca_dim = pca_dim

        self._split_type = split_type
        
        if apply_pca:
            self._pca = self.calc_pca(self._file['patches']['data'], self._pca_dim)

    def __len__(self):
        """ Return length of data set."""
        return self._samplelist.shape[0]

    def __getitem__(self, idx):
        """ 
        Return sample in data set at position 'idx'.

        Data patch and corresponding are loaded from hard disk. Transformations defined in
        `self._transform` are applied before returning the samples.

        Attributes:
        -----------
        idx : int
            Position of data in sample list.
        """
        
        if torch.is_tensor(idx):
            idx = idx.tolist()

        pos = self._samplelist[idx]

        patch = self._file['patches']['data'][pos]
        label = self._file['patches']['labels'][pos]
        
        # apply pca
        if self._apply_pca:
            shape = patch.shape
            patch = np.reshape(patch, (-1, patch.shape[2]))
            patch = self.pca.transform(patch)
            patch = np.reshape(patch, (shape[0], shape[1], self._pca_dim))
    
        sample = (patch, label)

        if self._transform:
            sample = self._transform(sample)

        return sample

    @staticmethod
    def calc_pca(patches, dim):
        """
        Calculates PCA to reduce the data's spectral dimensionality. Resulting PCA-object is
        returned. sklearn.decomposition.PCA is applied to data to reduce dimensionality. 
        """

        # get all samples
        X = []
        shape = patches[0].shape
        cx, cy = shape[0]//2, shape[1]//2
        for patch in patches:
           # because of patch sampling only center pixel is relevant
           # all other pixels are padding
           X.append(patch[cx,cy,:])
        
        # fit pca
        X = np.array(X)
        pca = PCA(n_components=dim, whiten=True)
        pca.fit(X)
        return pca
    
    @property
    def pca(self):
        """
        Returns pca object or None of if it does not exist.
        """
        return self._pca

    @pca.setter
    def pca(self, value):
        """
        Set pca Object.
        """
        self._pca = value
        if value is None:
            self._apply_pca = False
            self._pca_dim = None
        else : 
            self._apply_pca = True
            self._pca_dim = value.n_components
