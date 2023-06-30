#!/usr/bin/env/python
import os
import numpy as np
import tifffile
from typing import List, Any, Optional
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from hyperseg.datasets.transforms import ToTensor

class HSIRoad(pl.LightningDataModule):
    def __init__( 
            self,
            basepath: str,
            sensortype: str, # vis, nir, rgb
            batch_size: int,
            num_workers: int,
            precalc_histograms: bool=False,
            ):
        super().__init__()
        
        self.save_hyperparameters()

        self.basepath = Path(basepath)
        self.sensortype = sensortype
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transforms.Compose([
                            ToTensor()
                        ])
        self.precalc_histograms=precalc_histograms
        self.c_hist_train = None
        self.c_hist_val = None
        self.c_hist_test = None

        self.n_classes = 2

    def class_histograms(self):
        if self.c_hist_train is not None :
            return (self.c_hist_train, self.c_hist_val, self.c_hist_test)
        else :
            return None

    def setup(self, stage: Optional[str] = None):
        self.dataset_train = get_dataset(
                                data_dir=self.basepath,
                                sensortype=self.sensortype,
                                transform=self.transform,
                                mode='train')

        self.dataset_val = get_dataset(                                
                                data_dir=self.basepath, 
                                sensortype=self.sensortype, 
                                transform=self.transform,
                                mode='val')
        if self.precalc_histograms:
            self.c_hist_train = label_histogram(
                    self.dataset_train, self.n_classes)
            self.c_hist_val = label_histogram(
                    self.dataset_val, self.n_classes)

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
        # using val-set is not a typo, unfortunately the dataset authors provide only train and valid
        # ation sets. They use the validation set also as test set. We do the same to keep experiments
        # comparable
        return DataLoader(
                self.dataset_val, 
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers)



"""
layout 2.0 (33G raw data, 24G images, 4G masks)
    -hsi_road
         +images (3799 rgb, vis, nir tiff images in uint8, [c, h, w] format)
         +masks (3799 rgb, vis, nir tiff masks in uint8, [h, w] format)
         all.txt (serial number only)
         train.txt (serial number only)
         valid.txt (serial number only)
         vis_correction.txt (already applied)
         nir_correction.txt (already applied)

Based on: https://github.com/NUST-Machine-Intelligence-Laboratory/hsi_road/blob/master/datasets.py
"""

class HSIRoadDataset(Dataset):
    CLASSES = ('background', 'road')
    COLLECTION = ('rgb', 'vis', 'nir')

    def __init__(self, data_dir, collection, transform, classes=('background', 'road'), mode='train'):
        # 0 is background and 1 is road
        self.data_dir = data_dir
        self.collection = collection.lower()
        
        # mode == 'train' || mode == 'validation''
        path = os.path.join(data_dir, 'train.txt' if mode == 'train' else 'valid.txt')
        
        if mode == 'full':
            path = os.path.join(data_dir, 'all.txt')

        self.name_list = np.genfromtxt(path, dtype='str')
        self.classes = [self.CLASSES.index(cls.lower()) for cls in classes]
        self._transform = transform

    def __getitem__(self, i):
        # pick data
        name = '{}_{}.tif'.format(self.name_list[i], self.collection)

        image_path = os.path.join(self.data_dir, 'images', name)
        mask_path = os.path.join(self.data_dir, 'masks', name)
        image = tifffile.imread(image_path).astype(np.float32) / 255
        mask = tifffile.imread(mask_path).astype(np.long)

        sample = (image, mask)

        if self._transform:
            sample = self._transform(sample)
        return sample

    def __len__(self):
        return len(self.name_list)


def get_dataset(data_dir, sensortype, transform, mode):
    ds = None
    ds = HSIRoadDataset(data_dir=data_dir, collection=sensortype, transform=transform, mode=mode)
    return ds


