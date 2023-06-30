#!/usr/bin/env python

from .groundbased import HSDataModule
from torchvision import transforms
from hyperseg.datasets.transforms import ToTensor, PermuteData, ReplaceLabels

class HyKo2(HSDataModule):
    def __init__(self, label_set, **kwargs):
        super().__init__(**kwargs)
        if label_set == 'semantic':
            self.transform = transforms.Compose([
                ToTensor(),
                PermuteData(new_order=[2,0,1]),
                ReplaceLabels({0:10, 1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7, 9:8, 10:9}) # replace undefined label 0 with 10 and then shift labels by one
            ])
        elif label_set == 'material':
            self.transform = transforms.Compose([
                ToTensor(),
                PermuteData(new_order=[2,0,1]),
                ReplaceLabels({0:8, 1:0, 2:1, 3:2, 4:3, 5:4, 6:5, 7:6, 8:7}) # replace undefined label 0 with 8 and then shift labels by one
            ])
        else: 
            print('define labelset parameter as eiterh `semantic` or `material`')
            sys.exit()

