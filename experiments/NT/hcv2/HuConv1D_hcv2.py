from hsdatasets.groundbased.prep import apply_pca
from hsdatasets.groundbased import HyperspectralCityV2
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
    n_classes = 19
    n_channels = 10
    ignore_index = 19
    dataset_filepath = '/home/hyperseg/data/HyperspectralCityV2.h5'
    pca_out_filepath = f'/mnt/data/HyperspectralCityV2_PCA{n_channels}.h5'

    ### reduce dimensionality of dataset
    apply_pca(n_channels, dataset_filepath, pca_out_filepath)
    dataset_filepath = pca_out_filepath

    ## Training + Evaluation
    train_proportion = 0.5
    val_proportion = 0.1

    batch_size = 4
    num_workers = 14
    half_precision=True
    if half_precision:
        precision=16
    else:
        precision=32

    model_name = "HuConv1D"
    resume_path = None
    max_epochs = 600

    ## Logging
    dataset = "hcv2"
    modelname = "HuConv1D"

    log_dir = Path("/mnt/data/results").joinpath(dataset)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.fromtimestamp(time.time()).strftime('%Y%m%d-%H:%M:%S')

    data_module = HyperspectralCityV2(
            half_precision=half_precision,
            filepath=dataset_filepath, 
            num_workers=num_workers,
            batch_size=batch_size,
            train_prop=train_proportion,
            val_prop=val_proportion,
            n_classes=n_classes,
            manual_seed=manual_seed)

    model = HuConv1D(
            kernel_size=5,
            pool_size=2,
            n_channels=n_channels,
            n_classes=n_classes,
            label_def='/home/hyperseg/git/hsdatasets/labeldefs/HCv2_labels.txt', 
            loss_name='cross_entropy',
            learning_rate=0.001,
            optimizer_name='AdamW',
            optimizer_eps=1e-04,
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
