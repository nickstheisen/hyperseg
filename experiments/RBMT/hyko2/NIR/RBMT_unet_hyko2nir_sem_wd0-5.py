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
    n_classes = 10 # 11 - 1 because class 0 is undefined
    n_channels = 25
    ignore_index = 10

    ## Training + Evaluation
    train_proportion = 0.5
    val_proportion = 0.2
    batch_size = 16
    num_workers = 0
    half_precision=False
    if half_precision:
        precision=16
    else:
        precision=32
    log_dir = "/mnt/data/RBMT_results/hyko2NirSem"
    resume_path = None


    hyko2vissem_filepath = download_dataset('~/data','HyKo2-NIR_Semantic')
    data_module = HyKo2(
            filepath=hyko2vissem_filepath, 
            num_workers=num_workers,
            batch_size=batch_size,
            label_set='semantic',
            train_prop=train_proportion,
            val_prop=val_proportion,
            n_classes=n_classes,
            manual_seed=manual_seed)

    model = UNet(
            n_channels=n_channels,
            n_classes=n_classes,
            label_def='/home/hyperseg/data/hyko2_semantic_labels.txt', 
            loss_name='cross_entropy',
            learning_rate=0.01,
            optimizer_name='SGD',
            momentum=0.0,
            weight_decay=0.5,
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
            max_epochs=500,
            auto_lr_find=True,
            precision=precision,
            )
    
    # train model
    trainer.fit(model, 
            data_module, 
            ckpt_path=resume_path)