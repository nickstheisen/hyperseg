#!/usr/bin/env/python
import os
import numpy as np
from tqdm import tqdm
import tifffile
from typing import List, Any, Optional
from pathlib import Path

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from hyperseg.datasets.analysis.tools import StatCalculator
from hyperseg.datasets.transforms import ToTensor, Normalize, SpectralAverage

class HSIRoad(pl.LightningDataModule):
    def __init__( 
            self,
            basepath: str,
            sensortype: str, # vis, nir, rgb
            batch_size: int,
            num_workers: int,
            label_def: str,
            manual_seed: int=None,
            precalc_histograms: bool=False,
            normalize: bool=False,
            spectral_average: bool=False,
            prep_3dconv:bool=False,
            debug: bool = False,
            ):
        super().__init__()
        
        self.save_hyperparameters()

        self.basepath = Path(basepath)
        self.sensortype = sensortype
        self.debug = debug

        self.spectral_average = spectral_average
        self.prep_3dconv = prep_3dconv
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.label_def = label_def
        self.manual_seed = manual_seed

        self.transform = transforms.Compose([
                            ToTensor()
                        ])
        self.precalc_histograms=precalc_histograms
        self.c_hist_train = None
        self.c_hist_val = None
        self.c_hist_test = None

        #self.n_channels = 1 if self.spectral_average else 25 
        self.n_classes = 2
        self.undef_idx = -100


        if spectral_average:
            self.transform = transforms.Compose([
                                self.transform,
                                SpectralAverage()
                            ])

        if self.prep_3dconv:
            self.transform = transforms.Compose([
                self.transform,
                InsertEmptyChannelDim(1)
            ])

        # statistics (if normalization is activated)
        self.normalize = normalize
        self.means = None
        self.stds = None

        # read dimensions from image
        dataset = HSIRoadDataset(data_dir=self.basepath,
                    collection=self.sensortype,
                    transform=self.transform,
                    mode='train',
                    debug=self.debug)
        img, _ = dataset[0]
        img = img.squeeze()
        self.img_shape = img.shape[1:]
        self.n_channels = img.shape[0]
      
    def class_histograms(self):
        if self.c_hist_train is not None :
            return (self.c_hist_train, self.c_hist_val, self.c_hist_test)
        else :
            return None

    def setup(self, stage: Optional[str] = None):
        self.dataset_train = HSIRoadDataset(
                                data_dir=self.basepath,
                                collection=self.sensortype,
                                transform=self.transform,
                                mode='train',
                                debug=self.debug)

        self.dataset_val = HSIRoadDataset(                                
                                data_dir=self.basepath, 
                                collection=self.sensortype, 
                                transform=self.transform,
                                mode='val',
                                debug=self.debug)
        if self.precalc_histograms:
            self.c_hist_train = label_histogram(
                    self.dataset_train, self.n_classes)
            self.c_hist_val = label_histogram(
                    self.dataset_val, self.n_classes)

        # calculate data statistics for normalization
        if self.normalize:
            stat_calc = StatCalculator(self.dataset_train)
            self.means, self.stds = stat_calc.getDatasetStats()

            # enable normalization in whole data set
            self.dataset_train.enable_normalization(self.means, self.stds)
            self.dataset_val.enable_normalization(self.means, self.stds)

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
Free helper function to construct paths for HSIRoad-Dataset images and labels
"""
def get_file_path( basepath,
                    image_name,
                    collection,
                    image=True):
    file_name = '{}_{}.tif'.format(image_name, collection)
    if image:
        file_path = os.path.join(basepath, 'images', file_name)
    else:
        file_path = os.path.join(basepath, 'masks', file_name)

    return file_path

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

    def __init__(self, data_dir, collection, transform, classes=('background', 'road'), mode='train', debug=False):
        # 0 is background and 1 is road
        self.data_dir = data_dir
        self.collection = collection.lower()
        self.debug=debug
        
        # mode == 'train' || mode == 'validation''
        path = os.path.join(data_dir, 'train.txt' if mode == 'train' else 'valid.txt')
        
        if mode == 'full':
            path = os.path.join(data_dir, 'all.txt')

        self.name_list = np.genfromtxt(path, dtype='str')
        
        if self.debug:
            if mode =='train':
                self.name_list = self.name_list[:60]
            elif mode == 'val':
                self.name_list = self.name_list[:40]
            elif mode == 'full':
                self.name_list = self.name_list[:100]

        self.classes = [self.CLASSES.index(cls.lower()) for cls in classes]
        self._transform = transform

    def enable_normalization(self, means, stds):
        self._transform = transforms.Compose([
            self._transform,
            Normalize(means=means, stds=stds)
        ])
        self.mean = means
        self.std = stds

    def __getitem__(self, i):
        # pick data

        image_path = get_file_path( self.data_dir, self.name_list[i], self.collection, True)
        mask_path = get_file_path(self.data_dir, self.name_list[i], self.collection, False)
        image = tifffile.imread(image_path).astype(np.float32) / 255
        mask = tifffile.imread(mask_path).astype(int)

        sample = (image, mask)

        if self._transform:
            sample = self._transform(sample)
        return sample

    def __len__(self):
        return len(self.name_list)
