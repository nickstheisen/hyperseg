from hsdatasets.groundbased.prep import download_dataset
from hsdatasets.groundbased.groundbased import HyKo2
from hyperseg.models.imagebased import EffUNet
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from torch import nn

if __name__ == '__main__':

    seed_everything(42, workers=True)

    hyko2vissem_filepath = download_dataset('~/data','HyKo2-VIS_Semantic')
    data_module = HyKo2(filepath=hyko2vissem_filepath, batch_size=8)
    n_classes = 10 # 11 - 1 because class 0 is undefined
    ignore_index = 255


    model = EffUNet(
            n_channels=15,
            n_classes=n_classes,
            label_def='/home/nick/data/hyko2_semantic_labels.txt',
            loss_name='cross_entropy',
            learning_rate=0.001,
            optimizer_name='Adam',
            momentum=0.0,
            ignore_index=10,
            mdmc_average='samplewise',
            model='b0',
            dropout=0.1,
            freeze_backbone=False,
            pretrained=False)

    checkpoint_callback = ModelCheckpoint(
            monitor="val_iou_epoch",
            filename="checkpoint-effUNet-{epoch:02d}-{val_iou_epoch:.2f}",
            save_top_k=3,
            mode='max'
            )

    trainer = Trainer(
            gpus=1, 
            max_epochs=300,
            )
    trainer.fit(model, data_module)
    
