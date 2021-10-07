#!/usr/bin/env python

from hsdatasets.groundbased import HSDataModule
from hyperseg.models.imagebased import ImagebasedClassifier

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI

if __name__ == '__main__':
    cli = LightningCLI(
            model_class=ImagebasedClassifier, 
            datamodule_class=HSDataModule, 
            subclass_mode_model=True,
            subclass_mode_data=True,
            seed_everything_default=42)
