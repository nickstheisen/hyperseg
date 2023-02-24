from hsdatasets.groundbased.prep import download_dataset, apply_pca
from hsdatasets.groundbased.groundbased import HyperspectralCityV2
from hsdatasets.callbacks import ExportSplitCallback
from hyperseg.models.imagebased import UNetMod
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
    n_classes = 19 # 20 - 1 because class 255/19 is undefined
    n_channels = 1 # apply DR to reduce from 128 to X
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

    log_dir = "/mnt/data/RBMT_results/HCV2"
    resume_path = None

    data_module = HyperspectralCityV2(
            half_precision=half_precision,
            filepath=dataset_filepath, 
            num_workers=num_workers,
            batch_size=batch_size,
            train_prop=train_proportion,
            val_prop=val_proportion,
            n_classes=n_classes,
            manual_seed=manual_seed)

    model = UNetMod(
            n_channels=n_channels,
            n_classes=n_classes,
            label_def='/home/hyperseg/data/HCv2_labels.txt', 
            loss_name='cross_entropy',
            learning_rate=0.001,
            optimizer_name='AdamW',
            optimizer_eps=1e-04,
            momentum=0.0,
            weight_decay=0.0,
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
    export_split_callback = ExportSplitCallback()

    trainer = Trainer(
            default_root_dir=log_dir,
            callbacks=[checkpoint_callback, export_split_callback],
            accelerator='gpu',
            devices=[0], 
            max_epochs=300,
            auto_lr_find=True,
            precision=precision,
            )
    
    # train model
    trainer.fit(model, 
            data_module, 
            ckpt_path=resume_path)
