from hsdatasets.groundbased.prep import download_dataset
from hsdatasets.groundbased.groundbased import HSIRoad
from hsdatasets.callbacks import ExportSplitCallback
from hyperseg.models.imagebased import UNet
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
import torch

if __name__ == '__main__':
    manual_seed=42
    seed_everything(manual_seed, workers=True)

    # Parameters
    ## Data
    n_classes = 2
    n_channels = 25
    ignore_index = -100

    batch_size = 32
    num_workers = 8
    half_precision=False
    if half_precision:
        precision=16
    else:
        precision=32
    log_dir = "/mnt/data/RBMT_results/hsiroad"
    resume_path = None

    data_module = HSIRoad(
            basepath="/home/hyperseg/data/hsi_road/hsi_road",
            sensortype="nir",
            batch_size=batch_size,
            num_workers=num_workers)

    model = UNet(
            n_channels=n_channels,
            n_classes=n_classes,
            label_def="/home/hyperseg/data/hsi_road/hsi_road/hsi_road_label_def.txt",
            loss_name='cross_entropy',
            learning_rate=0.001,
            optimizer_name='AdamW',
            momentum=0.0,
            weight_decay=0.05,
            ignore_index=ignore_index,
            mdmc_average='samplewise',
            bilinear=True,
            class_weighting=None,
            batch_norm=False)

    checkpoint_callback = ModelCheckpoint(
            monitor="Validation/jaccard",
            filename="checkpoint-UNet-{epoch:02d}-{val_iou_epoch:.2f}",
            save_top_k=3,
            mode='max'
            )

    trainer = Trainer(
            default_root_dir=log_dir,
            callbacks=[checkpoint_callback],
            accelerator='gpu',
            devices=[0], 
            max_epochs=300,
            auto_lr_find=True,
            precision=precision,)
    
    # train model
    trainer.fit(model, 
            data_module, 
            ckpt_path=resume_path,
            )
