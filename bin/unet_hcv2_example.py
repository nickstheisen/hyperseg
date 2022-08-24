from hsdatasets.groundbased.prep import download_dataset, apply_pca
from hsdatasets.groundbased.groundbased import HyperspectralCityV2
from hyperseg.models.imagebased import UNet
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn
import torch

if __name__ == '__main__':
    manual_seed=42
    n_classes = 19 # 20 - 1 because class 255/19 is undefined
    n_channels = 25 # apply DR to reduce from 128 to 20
    ignore_index = 19
    batch_size = 1
    num_workers = 4
    half_precision=False
    if half_precision:
        precision=16
    else:
        precision=32


    seed_everything(manual_seed, workers=True)

    dataset_filepath = '/home/hyperseg/data/HyperspectralCityV2.h5'
    pca_out_filepath = f'/mnt/data/HyperspectralCityV2_PCA{n_channels}.h5'
    
    # reduce dimensionality of dataset
    apply_pca(n_channels, dataset_filepath, pca_out_filepath)

    dataset_filepath = pca_out_filepath

    #resume_path = '/home/hyperseg/data/results/HCV2/lightning_logs/version_1/checkpoints/checkpoint-UNet-epoch=61-val_iou_epoch=0.00.ckpt'
    resume_path = None

    data_module = HyperspectralCityV2(
            half_precision=half_precision,
            filepath=dataset_filepath, 
            num_workers=num_workers,
            batch_size=batch_size,
            train_prop=0.5,
            val_prop=0.1,
            n_classes=n_classes,
            manual_seed=manual_seed)

    model = UNet(
            n_channels=n_channels,
            n_classes=n_classes,
            label_def='/home/hyperseg/data/HCv2_labels.txt', 
            loss_name='cross_entropy',
            learning_rate=0.01,
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
            callbacks=[checkpoint_callback],
            accelerator='gpu',
            devices=[0], 
            max_epochs=200,
            auto_lr_find=True,
            precision=precision,
            )
    
    # train model
    trainer.fit(model, 
            data_module, 
            ckpt_path=resume_path)
