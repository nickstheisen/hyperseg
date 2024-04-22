#!/usr/bin/env python

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict

from hyperseg.datasets import get_datamodule
from hyperseg.models import get_model

from datetime import datetime
from pathlib import Path
import os

# set max_split_size_mb to 512 to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

valid_datasets = ['hsidrive','whuohs','hyko2', 'hsiroad', 'hcv2']
valid_models = ['unet', 'agunet', 'spectr']

def make_reproducible(manual_seed=42):
    seed_everything(manual_seed, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)

@hydra.main(version_base=None, config_path="conf", config_name="test_conf")
def test(cfg):
    print(OmegaConf.to_yaml(cfg))

    ## Logging
    log_dir = Path(cfg.logging.path+f"{cfg.logging.project_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    resume_path = cfg.training.resume_path
    loggers = []

    ts = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    pt = "" # full pretrained
    npt = "" # no pretrained backbone
    if cfg.model.name == 'deeplabv3plus':
        if cfg.model.pretrained_weights is not None:
            pt = "-PT"
        if not cfg.model.pretrained_backbone:
            npt = "-NPT"
    logname_run = f"test-{cfg.dataset.log_name}-{cfg.model.log_name}{pt}{npt}-{ts}"

    if cfg.logging.tb_logger:
        loggers.append(pl_loggers.TensorBoardLogger(
                version=logname_run,
                save_dir=log_dir,
        ))

    if cfg.logging.wb_logger:
        wandb.finish()
        loggers.append(pl_loggers.WandbLogger(
                name=logname_run,
                project=f"{cfg.logging.project_name}",
                save_dir=log_dir,
        ))
    

    ## Data Module
    if cfg.dataset.half_precision:
        precision="16-mixed"
    else:
        precision=32
    torch.set_float32_matmul_precision(cfg.training.mat_mul_precision)
    make_reproducible(cfg.training.seed)

    if cfg.dataset.pca is not None:
        pca_out_path = Path(cfg.dataset.pca_out_dir)
        
        if not pca_out_path.is_dir():
            raise RuntimeError("`pca_out_dir` must be a directory")
        
    datamodule = get_datamodule(cfg.dataset)

    ## Model
    with open_dict(cfg):
        cfg.model.n_channels = datamodule.n_channels
        cfg.model.n_classes = datamodule.n_classes
        cfg.model.spatial_size = list(datamodule.img_shape)
        cfg.model.ignore_index = datamodule.undef_idx
        cfg.model.label_def = datamodule.label_def

    model = get_model(cfg.model)


    ## Trainer
    trainer = Trainer(
            default_root_dir=log_dir,
            accelerator=cfg.training.accelerator,
            devices=cfg.training.devices, 
            precision=precision,
            logger=loggers,
            )
    trainer.test(model=model,
                datamodule=datamodule,
                ckpt_path=cfg.model.ckpt)

if __name__ == '__main__':
    test()
