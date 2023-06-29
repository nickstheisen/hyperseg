#!/usr/bin/env python

from torchvision import transforms
from hyperseg.datasets.transforms import ToTensor, PermuteData, ReplaceLabels
from .groundbased import HSDataModule

class HyperspectralCityV2(HSDataModule):
    def __init__(self, half_precision=False, **kwargs):
        super().__init__(**kwargs)
        self.half_precision = half_precision

        self.transform = transforms.Compose([
            ToTensor(half_precision=self.half_precision),
            PermuteData(new_order=[2,0,1]),
            ReplaceLabels({255:19})
        ])

