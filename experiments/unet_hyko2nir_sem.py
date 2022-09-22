from hsdatasets.groundbased.prep import download_dataset
from hsdatasets.groundbased.groundbased import HyKo2
from hyperseg.models.imagebased import UNet
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
import torch

if __name__ == '__main__':
    manual_seed=42
    n_classes = 10 # 11 - 1 because class 0 is undefined
    n_channels = 25
    ignore_index = 10
    batch_size = 32
    num_workers = 4
    half_precision=True
    if half_precision:
        precision=16
    else:
        precision=32
    log_dir = "~/data/results/hyko2NirSem"

    seed_everything(manual_seed, workers=True)

    resume_path = None

    hyko2vissem_filepath = download_dataset('~/data','HyKo2-NIR_Semantic')
    data_module = HyKo2(
            filepath=hyko2vissem_filepath, 
            num_workers=num_workers,
            batch_size=batch_size,
            label_set='semantic',
            train_prop=0.5,
            val_prop=0.1,
            n_classes=n_classes,
            manual_seed=manual_seed)

    model = UNet(
            n_channels=n_channels,
            n_classes=n_classes,
            label_def='/home/hyperseg/data/hyko2_semantic_labels.txt', 
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

    trainer = Trainer(
            default_root_dir=log_dir,
            callbacks=[checkpoint_callback],
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
