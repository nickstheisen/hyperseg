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
from hyperseg.models import get_model
from hyperseg.datasets.prep import apply_pca

from datetime import datetime
from pathlib import Path


valid_datasets = ['hsidrive','whuohs','hyko2', 'hsiroad', 'hcv2']
valid_models = ['unet', 'agunet', 'spectr']

def make_reproducible(manual_seed=42):
    seed_everything(manual_seed, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)

@hydra.main(version_base=None, config_path="conf", config_name="train_conf")
def train(cfg):
    print(OmegaConf.to_yaml(cfg))
    
    # Check Parameters
    if cfg.dataset.pca is not None:
        if cfg.dataset.pca_out_dir is None:
            raise RuntimeError("If `pca` is set, you also need to set `pca_out_dir`")
        if cfg.dataset.spectral_average is not None:
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
    log_dir = Path(cfg.logging.path+f"{cfg.logging.project_name}/{cfg.dataset.name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    resume_path = cfg.training.resume_path
    loggers = []

    ts = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    if cfg.logging.tb_logger:
        loggers.append(pl_loggers.TensorBoardLogger(
                version=f"{cfg.model.name}-{ts}",
                save_dir=log_dir,
        ))

    if cfg.logging.wb_logger:
        loggers.append(pl_loggers.WandbLogger(
                name=f"{cfg.model.name}-{ts}",
                project=f"{cfg.logging.project_name}-{cfg.dataset.name}",
                save_dir=log_dir,
        ))

    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            monitor="Validation/jaccard",
            filename="checkpoint-"+cfg.model.name+"-epoch-{epoch:02d}-val-iou-{Validation/jaccard:.3f}",
            auto_insert_metric_name=False,
            save_top_k=1,
            mode='max'
            )
    )
    callbacks.append(ModelSummary())
    if cfg.dataset.name not in ['whuohs']:
        callbacks.append(ExportSplitCallback()) # split is already defined in benchmark

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
