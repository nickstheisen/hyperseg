#!/usr/bin/env python

from .hsdataset import HSDataModule, HSDataset
from torchvision import transforms
from hyperseg.datasets.transforms import ToTensor, PermuteData, ReplaceLabels, SpectralAverage
from hyperseg.datasets.utils import apply_pca

from typing import List, Any, Optional

class HyKo2(HSDataModule):
    def __init__(self,  **kwargs):
        super().__init__(**kwargs)

        self.transform = transforms.Compose([
            ToTensor(),
            PermuteData(new_order=[2,0,1]),
        ])

        self.n_classes = 11
        self.undef_idx=0
        self.filepath_train = self.basepath.joinpath('hyko2_train.h5')
        self.filepath_test = self.basepath.joinpath('hyko2_test.h5')
        self.filepath_val = self.basepath.joinpath('hyko2_val.h5')

        if self.spectral_average:
            self.transform = transforms.Compose([
                self.transform,
                SpectralAverage()
            ])
        
        if self.pca is not None:
            self.enable_pca()

        dataset = HSDataset(self.filepath_val, transform=self.transform)
        img,_ = dataset[0]
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
        outpath_train = self.pca_out_dir.joinpath(f'hyko2_train_pca{self.pca}.h5')
        apply_pca(  self.pca, self.filepath_train, outpath_train, 
                    debug=self.debug, half_precision=False)
        self.filepath_train = outpath_train

        # test
        outpath_test = self.pca_out_dir.joinpath(f'hyko2_test_pca{self.pca}.h5')
        apply_pca(  self.pca, self.filepath_test, outpath_test, 
                    debug=self.debug, half_precision=False)
        self.filepath_test = outpath_test

        # val 
        outpath_val = self.pca_out_dir.joinpath(f'hyko2_val_pca{self.pca}.h5')
        apply_pca(  self.pca, self.filepath_val, outpath_val, 
                    debug=self.debug, half_precision=False)
        self.filepath_val = outpath_val
