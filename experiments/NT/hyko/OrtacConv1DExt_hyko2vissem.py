from hsdatasets.groundbased.prep import download_dataset
from hsdatasets.groundbased import HyKo2
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
    n_channels = 15
    ignore_index = 10

    ## Training + Evaluation
    train_proportion = 0.5
    val_proportion = 0.2
    batch_size = 16
    num_workers = 8
    half_precision=False
    if half_precision:
        precision=16
    else:
        precision=32

    model_name = "OrtacConv1DExt"
    resume_path = None
    max_epochs = 1000

    ## Logging
    dataset = "hyko2vissem"
    
    # train params
    learning_rate=0.01

    log_dir = Path("/mnt/data/results/").joinpath(dataset)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H:%M:%S')
    
    hyko2vissem_filepath = download_dataset('~/data', 'HyKo2-VIS_Semantic')
    data_module = HyKo2(
            filepath=hyko2vissem_filepath,
            num_workers=num_workers,
            batch_size=batch_size,
            label_set='semantic',
            train_prop=train_proportion,
            val_prop=val_proportion,
            n_classes=n_classes,
            manual_seed=manual_seed,
            precalc_histograms=False)

    model = OrtacConv1DExt(
            n_channels=n_channels,
            n_classes=n_classes,
            label_def='/home/hyperseg/git/hsdatasets/labeldefs/hyko2_semantic_labels.txt', 
            loss_name='cross_entropy',
            learning_rate=learning_rate,
            optimizer_name='Adam',
            momentum=0.0,
            weight_decay=0.0,
            ignore_index=ignore_index,
            mdmc_average='samplewise',
            class_weighting=None)

    checkpoint_callback = ModelCheckpoint(
            monitor="Validation/jaccard",
            filename="checkpoint-"+model_name+"-{epoch:02d}-{val_jaccard:.2f}",
            save_top_k=3,
            mode='max'
            )
    export_split_callback = ExportSplitCallback()

    tb_logger = TensorBoardLogger(save_dir=log_dir)

    wandb_logger = WandbLogger( save_dir=log_dir,
                                project=dataset,
                                name=f"{model_name}_SanityCheck"+timestamp)

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
