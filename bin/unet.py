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
    ignore_index = 255

    seed_everything(manual_seed, workers=True)

    hyko2vissem_filepath = download_dataset('~/data','HyKo2-VIS_Semantic')
    data_module = HyKo2(
            filepath=hyko2vissem_filepath, 
            num_workers=8,
            batch_size=8,
            train_prop=0.7,
            val_prop=0.1,
            n_classes=n_classes,
            manual_seed=manual_seed)

    model = UNet(
            n_channels=15,
            n_classes=n_classes,
            label_def='/home/nick/data/hyko2_semantic_labels.txt',
            loss_name='cross_entropy',
            learning_rate=0.001,
            optimizer_name='Adam',
            momentum=0.0,
            ignore_index=10,
            mdmc_average='samplewise',
            bilinear=True,
            class_weighting='ISNS')

    checkpoint_callback = ModelCheckpoint(
            monitor="Validation/jaccard",
            filename="checkpoint-UNet-{epoch:02d}-{val_iou_epoch:.2f}",
            save_top_k=3,
            mode='max'
            )

    trainer = Trainer(
            callbacks=[checkpoint_callback],
            gpus=[3], 
            max_epochs=600,
            auto_lr_find=True,
            )
    
    # train model
    trainer.fit(model, data_module)
