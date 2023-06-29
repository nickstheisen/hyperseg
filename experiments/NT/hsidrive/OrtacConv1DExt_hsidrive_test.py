from hsdatasets.groundbased.prep import download_dataset
from hsdatasets.groundbased.hsidrive import HSIDrive
from hsdatasets.callbacks import ExportSplitCallback
from hyperseg.models.spectrabased import OrtacConv1DExt
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from torch import nn
from pathlib import Path
import torch
import time
import datetime

if __name__ == '__main__':
    manual_seed=42
    seed_everything(manual_seed, workers=True)

    # Parameters
    ## Data
    n_classes = 10
    n_channels = 25
    ignore_index = 10

    ## Training + Evaluation
    train_proportion = 0.1
    val_proportion = 0.1
    batch_size = 8
    num_workers = 4
    half_precision=False
    if half_precision:
        precision=16
    else:
        precision=32

    model_name = "OrtacConv1DExt"
    max_epochs = 2000
    resume_path = None
    max_epochs = 4000
    resume_path = "/mnt/data/results/hsidrive/lightning_logs/version_33/checkpoints/checkpoint-OrtacConv1DExt-epoch=1745-jaccard_val=0.00.ckpt"

    ## Logging
    dataset = "hsidrive"

    log_dir = Path("/mnt/data/results/").joinpath(dataset)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H:%M:%S')

    data_module = HSIDrive(
            basepath = "/mnt/data/data/hsi-drive/Image_dataset",
            train_prop=train_proportion,
            val_prop=val_proportion,
            batch_size=batch_size, 
            num_workers=num_workers,
            precalc_histograms=False)

    model = OrtacConv1DExt(
            n_channels=n_channels,
            n_classes=n_classes,
            kernel_size=5,
            label_def='/home/hyperseg/git/hsdatasets/labeldefs/hsidrive-labels.txt', 
            loss_name='cross_entropy',
            learning_rate=0.0001,
            optimizer_name='Adam',
            momentum=0.0,
            weight_decay=0.0,
            ignore_index=ignore_index,
            mdmc_average='samplewise',
            class_weighting=None)

    checkpoint_callback = ModelCheckpoint(
            monitor="Validation/jaccard",
            filename="checkpoint-"+f"{model_name}"+"-{epoch:02d}-{jaccard_val:.2f}",
            save_top_k=3,
            mode='max'
            )
    export_split_callback = ExportSplitCallback()

    tb_logger = TensorBoardLogger(save_dir=log_dir)

    wandb_logger = WandbLogger( save_dir=log_dir,
                                project=dataset,
                                name=f"{model_name}_"+timestamp)

    trainer = Trainer(
            callbacks=[checkpoint_callback, export_split_callback],
            logger=[tb_logger, wandb_logger],
            accelerator='gpu',
            devices=[0], 
            max_epochs=max_epochs,
            auto_lr_find=True,
            precision=precision,
            )
    
    # train model
    trainer.fit(model, 
            data_module, 
            ckpt_path=resume_path)
