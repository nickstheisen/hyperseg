from hsdatasets.groundbased.prep import download_dataset
from hsdatasets.groundbased.hsidrive import HSIDrive
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
    n_classes = 10 
    n_channels = 25
    ignore_index = 10

    train_proportion = 0.6
    val_proportion = 0.2
    batch_size = 32
    num_workers = 4
    half_precision=False
    if half_precision:
        precision=16
    else:
        precision=32
    log_dir = "/mnt/data/RBMT_results/hsidrive"
    resume_path = None

    precalc_histograms = True


    data_module = HSIDrive(
            basepath = "/mnt/data/data/hsi-drive/Image_dataset",
            train_prop=train_proportion,
            val_prop=val_proportion,
            batch_size=batch_size, 
            num_workers=num_workers,
            precalc_histograms=precalc_histograms)
    data_module.setup()


    model = UNet(
            n_channels=n_channels,
            n_classes=n_classes,
            label_def="/home/hyperseg/git/hsdatasets/labeldefs/hsidrive-labels.txt",
            loss_name='cross_entropy',
            learning_rate=0.001,
            optimizer_name='AdamW',
            momentum=0.0,
            weight_decay=0.0,
            ignore_index=ignore_index,
            mdmc_average='samplewise',
            bilinear=True,
            class_weighting="ISNS",
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
            max_epochs=200,
            auto_lr_find=True,
            precision=precision,)
    
    # train model
    trainer.fit(model, 
            data_module, 
            ckpt_path=resume_path,
            )
