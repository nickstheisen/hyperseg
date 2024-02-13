#!/usr/bin/env python

from hyperseg.models import UNet, AGUNet, SpecTr

def get_model(cfg):
    if cfg.name == 'unet':
        model = UNet(
            n_channels=cfg.n_channels,
            n_classes=cfg.n_classes,
            ignore_index=cfg.ignore_index,
            label_def=cfg.label_def,
            loss_name=cfg.loss_name,
            learning_rate=cfg.learning_rate,
            optimizer_name=cfg.optimizer_name,
            optimizer_eps=cfg.optimizer_eps,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            bilinear=cfg.bilinear,
            batch_norm=cfg.batch_norm,
            class_weighting=cfg.class_weighting,
            export_preds_every_n_epochs=cfg.export_preds_every_n_epochs,
            da_hflip=cfg.da_hflip,
            dropout=cfg.dropout,
        )
    elif cfg.name == 'agunet':
        model = AGUNet(
            n_channels=cfg.n_channels,
            n_classes=cfg.n_classes,
            label_def=cfg.label_def,
            ignore_index=cfg.ignore_index,
            loss_name=cfg.loss_name,
            learning_rate=cfg.learning_rate,
            optimizer_name=cfg.optimizer_name,
            optimizer_eps=cfg.optimizer_eps,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            bilinear=cfg.bilinear,
            batch_norm=cfg.batch_norm,
            class_weighting=cfg.class_weighting,
            export_preds_every_n_epochs=cfg.export_preds_every_n_epochs,
            da_hflip=cfg.da_hflip,
        )
    elif cfg.name == 'spectr':
        model = SpecTr(
            n_channels=cfg.n_channels,
            n_classes=cfg.n_classes,
            label_def=cfg.label_def,
            ignore_index=cfg.ignore_index,
            loss_name=cfg.loss_name,
            learning_rate=cfg.learning_rate,
            optimizer_name=cfg.optimizer_name,
            optimizer_eps=cfg.optimizer_eps,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            class_weighting=cfg.class_weighting,
            export_preds_every_n_epochs=cfg.export_preds_every_n_epochs,
            da_hflip=cfg.da_hflip,
            spatial_size=cfg.spatial_size,
            use_entmax15=cfg.use_entmax15,            
        )

    else:
        raise NotImplementedError(f"Model '{cfg.name}' does not exist.")
    
    return model
