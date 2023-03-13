from hsdatasets.groundbased.prep import download_dataset
from hsdatasets.groundbased import HSIRoad
from hsdatasets.callbacks import ExportSplitCallback
from hyperseg.models.spectrabased import HuConv1D
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
    n_classes = 2
    n_channels = 25
    ignore_index = -100

    ## Training + Evaluation
    batch_size = 32
    num_workers = 8
    half_precision=False
    if half_precision:
        precision=16
    else:
        precision=32

    model_name = "HuConv1D"
    resume_path = None
    max_epochs = 1000

    ## Logging
    dataset = "hsiroad"
    modelname = "HuConv1D"

    log_dir = Path("/mnt/data/results").joinpath(dataset)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H:%M:%S')

    data_module = HSIRoad(
            basepath = "/home/hyperseg/data/hsi_road/hsi_road",
            sensortype="nir",
            batch_size=batch_size, 
            num_workers=num_workers)

    model = HuConv1D(
            kernel_size=5,
            pool_size=2,
            n_channels=n_channels,
            n_classes=n_classes,
            label_def='/home/hyperseg/git/hsdatasets/labeldefs/hsi_road_label_def.txt', 
            loss_name='cross_entropy',
            learning_rate=0.001,
            optimizer_name='Adam',
            momentum=0.0,
            weight_decay=0.0,
            ignore_index=ignore_index,
            mdmc_average='samplewise',
            class_weighting=None)

    checkpoint_callback = ModelCheckpoint(
            monitor="Validation/jaccard",
            filename="checkpoint-"+f"{modelname}"+"-{epoch:02d}-{jaccard_val:.2f}",
            save_top_k=3,
            mode='max'
            )

    tb_logger = TensorBoardLogger(save_dir=log_dir)

    wandb_logger = WandbLogger( save_dir=log_dir,
                                project=dataset,
                                name=f"{modelname}_"+timestamp)

    trainer = Trainer(
            callbacks=[checkpoint_callback],
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
