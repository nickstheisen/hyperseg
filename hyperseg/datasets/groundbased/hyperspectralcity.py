#!/usr/bin/env python

from torchvision import transforms
from hyperseg.datasets.transforms import ToTensor, PermuteData, ReplaceLabels, SpectralAverage, InsertEmptyChannelDim
from .groundbased import HSDataModule, GroundBasedHSDataset

class HyperspectralCityV2(HSDataModule):
    def __init__(self, half_precision=False, n_pc=None, **kwargs):
        super().__init__(**kwargs)
        self.half_precision = half_precision
        self.n_pc = n_pc

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
        if self.prep_3dconv:
            self.transform = transforms.Compose([
                self.transform,
                InsertEmptyChannelDim(1)
            ])

#        if self.spectral_average:
#            self.n_channels = 1
#        elif self.n_pc is not None:
#            self.n_channels = self.n_pc
#        else:
#            self.n_channels = 128
#
        self.n_classes = 19
        self.undef_idx=19

        dataset = GroundBasedHSDataset(self.filepath, transform=self.transform)
        img,_ = dataset[0]
        img = img.squeeze()
        self.img_shape = img.shape[1:]
        self.n_channels = img.shape[0]

