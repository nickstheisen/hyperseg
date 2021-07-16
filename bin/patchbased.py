#!/usr/bin/env python

from pathlib import Path

from hsdatasets.transforms import ToTensor, InsertEmptyChannelDim, PermuteData
from hsdatasets.remotesensing import RSDataset
from hyperseg.models.patchbased import PatchbasedClassifier

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

import pytorch_lightning as pl
from pytorch_lightning.utilities.cli import LightningCLI

import argparse

if __name__ == '__main__':
    cli = LightningCLI(PatchbasedClassifier, 
            RSDataset, 
            seed_everything_default=42)
