from hsdatasets.groundbased.prep import download_dataset
from hsdatasets.groundbased.groundbased import HyKo2
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
    n_classes = 8 # 9 - 1 because class 0 is undefined
    n_channels = 25
    ignore_index = 8

    ## Training + Evaluation
    train_proportion = 0.5
    val_proportion = 0.2
    batch_size = 32
    num_workers = 4
    half_precision=False
    if half_precision:
        precision=16
    else:
        precision=32
    log_dir = "~/data/results/hyko2NirMat"
    resume_path = None


    hyko2vissem_filepath = download_dataset('~/data','HyKo2-NIR_Material')
    data_module = HyKo2(
            filepath=hyko2vissem_filepath, 
            num_workers=num_workers,
            batch_size=batch_size,
            label_set='material',
            train_prop=train_proportion,
            val_prop=val_proportion,
            n_classes=n_classes,
            manual_seed=manual_seed)

    model = UNet(
            n_channels=n_channels,
            n_classes=n_classes,
            label_def='/home/hyperseg/data/hyko2_material_labels.txt', 
            loss_name='cross_entropy',
            learning_rate=0.001,
            optimizer_name='Adam',
            momentum=0.0,
            ignore_index=ignore_index,
            mdmc_average='samplewise',
            bilinear=True,
            class_weighting=None)

    checkpoint_callback = ModelCheckpoint(
            monitor="Validation/jaccard",
            filename="checkpoint-UNet-{epoch:02d}-{val_iou_epoch:.2f}",
            save_top_k=3,
            mode='max'
            )
    export_split_callback = ExportSplitCallback()

    trainer = Trainer(
            default_root_dir=log_dir,
            callbacks=[checkpoint_callback, export_split_callback],
            accelerator='gpu',
            devices=[0], 
            max_epochs=600,
            auto_lr_find=True,
            precision=precision,
            )
    
    # train model
    trainer.fit(model, 
            data_module, 
            ckpt_path=resume_path)
