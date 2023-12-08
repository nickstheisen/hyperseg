#!/usr/bin/env python

from .hsdataset import HSDataModule, HSDataset
from torchvision import transforms
from hyperseg.datasets.transforms import ToTensor, PermuteData, ReplaceLabels, SpectralAverage

class HyKo2(HSDataModule):
    def __init__(self, label_set, **kwargs):
        super().__init__(**kwargs)
        self.label_set = label_set
        if self.label_set == 'semantic':
            self.transform = transforms.Compose([
                ToTensor(),
                PermuteData(new_order=[2,0,1]),
            ])
            self.n_classes = 11
            self.undef_idx=0

        elif self.label_set == 'material':
            self.transform = transforms.Compose([
                ToTensor(),
                PermuteData(new_order=[2,0,1]),
            ])
            self.n_classes = 9
            self.undef_idx=0

        else: 
            print('define labelset parameter as either `semantic` or `material`')
            sys.exit()

        if self.spectral_average:
            self.transform = transforms.Compose([
                self.transform,
                SpectralAverage()
            ])


        dataset = HSDataset(self.filepath, transform=self.transform)
        img,_ = dataset[0]
        img = img.squeeze()
        self.img_shape = img.shape[1:]
        self.n_channels = img.shape[0]

