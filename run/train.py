#!/usr/bin/env python

import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary
from torchsummary import summary

from hyperseg.datasets import get_datamodule
from hyperseg.datasets.callbacks import ExportSplitCallback
from hyperseg.models.models import get_model
from hyperseg.datasets.prep import apply_pca

from datetime import datetime
from argparse import ArgumentParser
from pathlib import Path


valid_datasets = ['hsidrive','whuohs','hyko2', 'hsiroad', 'hcv2']
valid_models = ['unet', 'agunet', 'spectr']

def make_reproducible(manual_seed=42):
    seed_everything(manual_seed, workers=True)
    torch.use_deterministic_algorithms(True)

def train(args):
    print(args)
    
    # Check Parameters
    if args.pca is not None:
        if args.pca_out_dir is None:
            raise RuntimeError("If --pca is set, you also need to set --pca-out-dir")
        if args.spectral_average is not None:
            raise RuntimeError("--pca and --spectral-average can not be used at the same time.")

    # Training Configuration

    ## General
    torch.set_float32_matmul_precision('high') # can be set to 'medium'|'high'|'highest'
    make_reproducible()
    
    ## Logging
    log_dir = Path(args.log_dir+f"{args.project_name}/{args.dataset_name}")
    log_dir.mkdir(parents=True, exist_ok=True)
    resume_path = args.resume_path
    loggers = []

    ts = datetime.now().strftime("%Y%m%d_%H-%M-%S")
    if args.tb_logger:
        loggers.append(pl_loggers.TensorBoardLogger(
                version=f"{args.model_name}-{ts}",
                save_dir=log_dir,
        ))

    if args.wb_logger:
        loggers.append(pl_loggers.WandbLogger(
                name=f"{args.model_name}-{ts}",
                project=f"{args.project_name}-{args.dataset_name}",
                save_dir=log_dir,
        ))

    callbacks = []
    callbacks.append(
        ModelCheckpoint(
            monitor="Validation/jaccard",
            filename="checkpoint-"+args.model_name+"-epoch-{epoch:02d}-val-iou-{Validation/jaccard:.3f}",
            auto_insert_metric_name=False,
            save_top_k=1,
            mode='max'
            )
    )
    callbacks.append(ModelSummary())
    if args.dataset_name not in ['whuohs']:
        callbacks.append(ExportSplitCallback())

    ## Data Module
    if args.half_precision:
        precision="16-mixed"
    else:
        precision=32

    if args.pca is not None:
        pca_out_path = Path(args.pca_out_dir)
        
        if not pca_out_path.is_dir():
            raise RuntimeError("--pca-out-dir must be a directory")
        
        pca_out_path = pca_out_path.joinpath(f"{args.dataset_name}_PCA{args.pca}")
        apply_pca(args.pca, args.dataset_basepath, pca_out_path)
        args.dataset_basepath = pca_out_path

    datamodule = get_datamodule(args.dataset_name, 
                basepath=args.dataset_basepath,
                batch_size=args.batch_size, 
                spectral_average=args.spectral_average,
                prep_3dconv=args.prep_3dconv,
                debug=args.debug)

    ## Model
    model = get_model(args.model_name, 
                n_channels=datamodule.n_channels, 
                n_classes=datamodule.n_classes,
                spatial_size=datamodule.img_shape,
                ignore_index=datamodule.undef_idx,
                label_def=datamodule.label_def,
                use_entmax15="softmax",
                bilinear=False, # bilinear Upsampling with pytorch is non-deterministic on GPU
                )
    
    ## Misc
    if args.compile:
        model = torch.compile(model)

    ## Trainer
    trainer = Trainer(
            default_root_dir=log_dir,
            callbacks=callbacks,
            accelerator='gpu',
            devices=[0], 
            max_epochs=args.max_epochs,
            precision=precision,
            enable_model_summary=False, # enable for default model parameter printing at start
            logger=loggers,
            )

    # Train!
    trainer.fit(model, 
            datamodule, 
            ckpt_path=args.resume_path)

if __name__ == '__main__':

    parser = ArgumentParser(
                prog='train',
                description = ('Start training of semantic segmentation model for hyperspectral image'
                                ' datasets.')
            )

    parser.add_argument('dataset_name', choices=valid_datasets, help="name of dataset")
    parser.add_argument('model_name', choices=valid_models, help="name of model")
    parser.add_argument('project_name', help="name of project (arbitrary)")
    parser.add_argument('dataset_basepath', help="path to dataset folder")
    parser.add_argument('-l', '--log_dir', default="/mnt/data/logs/", 
        help="directory where logs are stored")
    parser.add_argument('-r', '--resume_path', default=None, 
        help="path to checkpoint to continue training")
    parser.add_argument('-p', '--half_precision', action='store_true',
        help="if set half precision training is used")
    parser.add_argument('-e', '--max_epochs', default=500,
        help="max. Training epochs", type=int)
    parser.add_argument('-t', '--tb_logger', action='store_true',
        help="if set logs for tensorboard are exported")
    parser.add_argument('-w', '--wb_logger', action='store_true',
        help="if set logs for weights & biases are exported (must be installed and configured)")
    parser.add_argument('-c', '--compile', action='store_true',
        help=("if set the model is compiled before training (only possible with pytorch >2.x)"
            " [Does not work yet]"))
    parser.add_argument('-b', '--batch_size', default=32, help='defines the batch size', 
            type=int)
    parser.add_argument('-a', '--spectral_average', action='store_true',
        help=("Averages over spectral dim. to reduce input dim. to 1."))
    parser.add_argument('--pca', type=int, 
        help=("Applies PCA and reduces data to n PC's (may require some time)."))
    parser.add_argument('--pca-out-dir', type=str,
        help=("Defines path where result of pca should be stored."))
    parser.add_argument('--prep-3dconv', action='store_true',
        help=("If set, data is prepared to be used with 3D-convs."))
    parser.add_argument('--debug', action='store_true',
        help=("Reduces the datasets to 100 samples overall to allow faster debugging."))

    args = parser.parse_args()
    train(args)
