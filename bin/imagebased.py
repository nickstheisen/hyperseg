#!/usr/bin/env python

from hsdatasets.groundbased import HSDataModule
from hyperseg.models.imagebased import ImagebasedClassifier

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI

if __name__ == '__main__':
    cli = LightningCLI(ImagebasedClassifier, 
            HSDataModule, 
            seed_everything_default=42)
