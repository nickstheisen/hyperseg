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
from hyperseg.datasets.transforms import ToTensor, Normalize

class HSIRoad(pl.LightningDataModule):
    def __init__( 
            self,
            basepath: str,
            sensortype: str, # vis, nir, rgb
            batch_size: int,
            num_workers: int,
            precalc_histograms: bool=False,
            normalize: bool=False,
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

        # statistics (if normalization is activated)
        self.normalize = normalize
        self.means = None
        self.stds = None
       
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
                                mode='train')

        self.dataset_val = HSIRoadDataset(                                
                                data_dir=self.basepath, 
                                collection=self.sensortype, 
                                transform=self.transform,
                                mode='val')
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

"""
Code below did not work. Viewing each pixel as sample is computationally to expensive to 
do realistically. Also using them later for training, a Batch with 100 samples may stem from 100 different images which would lead to immense access times. 
Only indexing the data-set, in a sense that we compile a list that contains tuples (name of image, y-position of patch-center, x-position of patch center) takes 2h+ and requires around 20GB of RAM after around 35%.

class HSIRoadPatchesDataset(Dataset):
    CLASSES = ('background', 'road')
    COLLECTION = ('rgb', 'vis', 'nir')

    def __init__(   self,
                    data_dir, 
                    collection,
                    transform,
                    patch_size,
                    classes=('background','road'),
                    mode='train'):
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
        
        # load single image to get dimension
        image_path = get_file_path( self.data_dir, self.name_list[0], self.collection, True)
        image = tifffile.imread(image_path)
        self.dims, self.height, self.width = image.shape

        self.patch_size = patch_size
        self.patch_radius = patch_size // 2
       
        # construct mapping from id -> imagepixel
        self.sample_list, self.num_samples = self.construct_sample_list()
        print(self.num_samples)
        for i in range(10):
            print(self.sample_list[i])


    def construct_sample_list(self):
        sample_list = None
        
        # we can do this outside of the loop because the image dimension is the same
        # for all images in the data set
        ## Do not consider border pixels to avoid padded samples
        ## repeat all elements from 0-`height` `width` times (0,..,0,1,...
        y_indices = np.repeat(np.arange(
                                self.patch_radius, 
                                self.height - self.patch_radius
                            ), self.width - 2*self.patch_radius )
        ## repeat 0-`width` `height` times
        x_indices = np.tile(np.arange(
                                self.patch_radius,
                                self.width - self.patch_radius
                            ), self.height- 2*self.patch_radius)

        for name in tqdm(self.name_list):
            name_rep = np.repeat(name, 
                        (self.width-2*self.patch_radius)*(self.height-2*self.patch_radius))
            image_samples = np.stack((name_rep, x_indices, y_indices))
            
            # construct tuple with img-name and pixel-position
            ## It is not necessary to consider 'undefined'-pixels as they are not apparent in this
            ## dataset. There is only 'road' and 'no road'
            if sample_list is None:
                sample_list = image_samples
            else:
                sample_list = np.concatenate((sample_list, image_samples))
        return sample_list, sample_list.size
    

    def __getitem__(self, i):
        # pick data
        pass

    def __len__(self):
        pass
"""
