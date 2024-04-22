#!/usr/bin/env python

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torchsummary import summary

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import wandb

from hyperseg.datasets import get_datamodule
from hyperseg.datasets.callbacks import ExportSplitCallback
from hyperseg.models import get_model

from datetime import datetime
from pathlib import Path
import os

# set max_split_size_mb to 512 to avoid fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"


valid_datasets = ['hsidrive','whuohs','hyko2', 'hsiroad', 'hcv2']
valid_models = ['unet', 'agunet', 'spectr', 'deeplabv3plus']

def make_reproducible(manual_seed=42):
    seed_everything(manual_seed, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)

@hydra.main(version_base=None, config_path="conf", config_name="train_conf")
def train(cfg):
    print(OmegaConf.to_yaml(cfg))
    
    if cfg.dataset.pca is not None:
        if cfg.dataset.pca_out_dir is None:
            raise RuntimeError("If `pca` is set, you also need to set `pca_out_dir`")
        if cfg.dataset.spectral_average == True:
            raise RuntimeError("`pca` and `spectral_average` can not be used at the same time.")

    if cfg.model.name not in valid_models:
        raise RuntimeError(f"Invalid model. (Options: {valid_models}")
    if cfg.dataset.name not in valid_datasets:
        raise RuntimeError(f"Invalid dataset. (Options: {valid_datasets}")

    # Training Configuration

    ## General
    torch.set_float32_matmul_precision(cfg.training.mat_mul_precision) 
    make_reproducible(cfg.training.seed)

    
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
    logname_run = f"{cfg.dataset.log_name}-{cfg.model.log_name}{pt}{npt}-{ts}"

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

    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            monitor="Validation/jaccard",
            filename="checkpoint-"+cfg.model.log_name+"-epoch-{epoch:02d}-val-iou-{Validation/jaccard:.3f}",
            auto_insert_metric_name=False,
            save_top_k=1,
            mode='max'
            )
    )
    callbacks.append(ModelSummary())

    if cfg.training.early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="Validation/jaccard",
                patience=15,
                mode="max"
            )
        )

    ## Data Module
    if cfg.dataset.half_precision:
        precision="16-mixed"
    else:
        precision=32

    datamodule = get_datamodule(cfg.dataset)

    ## Model
    with open_dict(cfg):
        cfg.model.n_channels = datamodule.n_channels
        cfg.model.n_classes = datamodule.n_classes
        cfg.model.spatial_size = list(datamodule.img_shape)
        cfg.model.ignore_index = datamodule.undef_idx
        cfg.model.label_def = datamodule.label_def

    print("Channels: ", cfg.model.n_channels)

    model = get_model(cfg.model)
    ## Misc
    if cfg.model.compile:
        model = torch.compile(model)

    ## Trainer
    trainer = Trainer(
            default_root_dir=log_dir,
            callbacks=callbacks,
            accelerator=cfg.training.accelerator,
            devices=cfg.training.devices, 
            max_epochs=cfg.training.max_epochs,
            precision=precision,
            enable_model_summary=False, # enable for default model parameter printing at start
            logger=loggers,
            )

    # Train!
    trainer.fit(model, 
            datamodule, 
            ckpt_path=cfg.training.resume_path)

if __name__ == '__main__':
    train()
