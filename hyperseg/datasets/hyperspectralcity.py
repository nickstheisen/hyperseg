#!/usr/bin/env python

from pathlib import Path
import numpy as np
from typing import List, Any, Optional
from imageio import imread

from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
import torch

from hyperseg.datasets.transforms import ToTensor, PermuteData, ReplaceLabels, SpectralAverage, InsertEmptyChannelDim
from hyperseg.datasets.utils import label_histogram
from .hsdataset import HSDataModule, HSDataset

N_DEBUG_SAMPLES = 5

class HyperspectralCityV2(pl.LightningDataModule):

    def __init__(
            self,
            basepath: str,
            num_workers: int,
            batch_size: int,
            label_def: str,
            manual_seed: int=None,
            precalc_histograms: bool=False,
            normalize: bool=False,
            spectral_average: bool=False,
            debug: bool = False,
            half_precision=False,
    ):
        super().__init__()
        
        self.save_hyperparameters()

        self.basepath = basepath
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.manual_seed = manual_seed
        self.debug = debug
        
        self.label_def = label_def

        self.spectral_average = spectral_average
        
        self.precalc_histograms=precalc_histograms
        self.c_hist_train = None
        self.c_hist_val = None
        self.c_hist_test = None

        # statistics (if normalization is activated)
        self.normalize = normalize
        self.means = None
        self.stds = None

        self.half_precision = half_precision

        self.transform = transforms.Compose([
            ToTensor(half_precision=self.half_precision),
            PermuteData(new_order=[2,0,1]),
            ReplaceLabels({255:19})
        ])

        if self.spectral_average:
            self.transform = transforms.Compose([
                self.transform,
                SpectralAverage()
            ])

        self.n_classes = 20
        self.undef_idx=19
        
        dataset = HCV2Dataset(self.basepath, transform=self.transform, mode='test')
        img,_ = dataset[0]
        self.img_shape = img.shape[1:]
        self.n_channels = img.shape[0]

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
        ## TODO
        dataset = HCV2Dataset(self.basepath, 
            transform=self.transform, 
            debug=self.debug, 
            mode='train')

        # Train-test-split is 80/20. From those 80% we use 10% for validation and 90% for training
        train_size = round(0.9 * len(dataset))
        val_size = len(dataset) - train_size

        if self.manual_seed is not None:
            self.dataset_train, self.dataset_val = random_split(
                    dataset, 
                    [train_size, val_size], 
                    generator=torch.Generator().manual_seed(self.manual_seed))
        else :
            self.dataset_train, self.dataset_val = random_split(
                    dataset, 
                    [train_size, val_size])
        self.dataset_test = HCV2Dataset(self.basepath,
             transform=self.transform,
             debug=self.debug,
             mode='test')

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


class HCV2Dataset(Dataset):

    def __init__(self, basepath, transform, debug=False, mode='train'):
        self._basepath = Path(basepath)
        if mode == 'train':
            self._basepath = self._basepath.joinpath('train')
        elif mode == 'test':
            self._basepath = self._basepath.joinpath('test')

        self._imagedir = self._basepath.joinpath('images')
        self._labeldir = self._basepath.joinpath('labels') 

        self._samplelist = [img.stem for img in self._imagedir.glob('*.npy')]
        
        self.debug = debug
        self._transform = transform

        if self.debug:
            # only use N_DEBUG_SAMPLES samples overall
            self._samplelist = self._samplelist[:N_DEBUG_SAMPLES]

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
        image_path = self._imagedir.joinpath(f'{self._samplelist[idx]}.npy')
        label_path = self._labeldir.joinpath(f'rgb{self._samplelist[idx]}_gray.png')
        sample = (np.load(image_path), imread(label_path))

        if self._transform:
            sample = self._transform(sample)
        
        return sample
