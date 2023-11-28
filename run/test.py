#!/usr/bin/env python

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from torchsummary import summary

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from hyperseg.datasets import get_datamodule
from hyperseg.datasets.callbacks import ExportSplitCallback
from hyperseg.models import load_model
from hyperseg.datasets.prep import apply_pca

from datetime import datetime
from pathlib import Path


valid_datasets = ['hsidrive','whuohs','hyko2', 'hsiroad', 'hcv2']
valid_models = ['unet', 'agunet', 'spectr']

def make_reproducible(manual_seed=42):
    seed_everything(manual_seed, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)

@hydra.main(version_base=None, config_path="conf", config_name="test_conf")
def test(cfg):
    ## Data Module
    if cfg.dataset.half_precision:
        precision="16-mixed"
    else:
        precision=32

    if cfg.dataset.pca is not None:
        pca_out_path = Path(cfg.dataset.pca_out_dir)
        
        if not pca_out_path.is_dir():
            raise RuntimeError("`pca_out_dir` must be a directory")
        
        pca_out_path = pca_out_path.joinpath(f"{cfg.dataset.name}_PCA{cfg.dataset.pca}")
        apply_pca(cfg.dataset.pca, cfg.dataset.basepath, pca_out_path)
        cfg.dataset.basepath = pca_out_path
        
    datamodule = get_datamodule(cfg.dataset)

    ## Model
    with open_dict(cfg):
        cfg.model.n_channels = datamodule.n_channels
        cfg.model.n_classes = datamodule.n_classes
        cfg.model.spatial_size = list(datamodule.img_shape)
        cfg.model.ignore_index = datamodule.undef_idx
        cfg.model.label_def = datamodule.label_def

    model = load_model(cfg.model)

    ## Trainer
    trainer = Trainer(
            accelerator=cfg.training.accelerator,
            devices=cfg.training.devices, 
            precision=precision,
            )
    trainer.test(model,
            datamodule.test)

if __name__ == '__main__':
    test()
