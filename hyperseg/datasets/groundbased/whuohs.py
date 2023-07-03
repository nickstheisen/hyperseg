#!/usr/bin/env python

from pathlib import Path
#from osgeo import gdal
import tifffile
import numpy as np
from torchvision import transforms
from typing import List, Any, Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from hyperseg.datasets.transforms import ToTensor

class WHUOHS(pl.LightningDataModule):
    def __init__( 
            self,
            basepath: str,
            batch_size: int,
            num_workers: int,
            ):
        super().__init__()
        
        self.save_hyperparameters()

        self.basepath = Path(basepath).expanduser()
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
                            ToTensor()
                        ])
        self.c_hist_train = None
        self.c_hist_val = None
        self.c_hist_test = None

        self.n_classes = 24

    def setup(self, stage: Optional[str] = None):
        self.dataset_train = WHUOHSDataset(
                                basepath=self.basepath,
                                transform=self.transform,
                                mode='train')

        self.dataset_val = WHUOHSDataset(
                                basepath=self.basepath,
                                transform=self.transform,
                                mode='val')
        
        self.dataset_test = WHUOHSDataset(
                                basepath=self.basepath,
                                transform=self.transform,
                                mode='test')

    def train_dataloader(self):
        return DataLoader(
                self.dataset_train,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers)
    def val_dataloader(self):
        return DataLoader(
                self.dataset_val,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(
                self.dataset_val, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)

class WHUOHSDataset(Dataset):
    def __init__(self, 
                basepath,
                transform,
                mode='train'):
        self.basepath = Path(basepath).expanduser()
        self._transform = transform
        self.mode = mode

        if mode not in ('train','test','val'):
            raise RuntimeError("Invalid mode! It must be `train`,`test` or `val`.")
        
        self.imagedir = self.basepath.joinpath(self.mode,'image')
        self.labeldir = self.basepath.joinpath(self.mode,'label')
        self.namelist = [ p.stem for p in self.imagedir.iterdir() 
                            if(p.suffix == '.tif') ]

    def __getitem__(self, i):
        image_path = self.imagedir.joinpath(f'{self.namelist[i]}.tif')
        label_path = self.labeldir.joinpath(f'{self.namelist[i]}.tif')
        #image = gdal.Open(image_path, gdal.GA_ReadOnly)
        #label = gdal.Open(label_path, gdal.GA_ReadOnly)

        #image = image.ReadAsArray().astype(np.float32) / 10000.0
        #label = label.ReadAsArray().astype(np.longlong)

        ''' 
        Above you see how the data is loaded in the authors repo:
        https://github.com/zjjerica/WHU-OHS-Pytorch/tree/main
        However, setting up GDAL with python is a pain in the ass and has dependencies that
        cannot be installed with pip. Therefore we use tifffile-package instead.
        '''
        image = tifffile.imread(image_path).astype(np.float32) / 10000.0
        label = tifffile.imread(label_path).astype(np.longlong)

        sample = (image, label)

        if self._transform:
            sample = self._transform(sample)
        return sample

    def __len__(self):
        return len(self.namelist)

if __name__ == '__main__':
    datamodule = WHUOHS(basepath='/mnt/data/data/WHU-OHS/',
                        batch_size=8,
                        num_workers=8)
    datamodule.setup()
    print(len(datamodule.test_dataloader()))
    for data in datamodule.test_dataloader():
        print(data[0].max(), data[0].min())
    print(len(datamodule.train_dataloader()))
    print(len(datamodule.val_dataloader()))




