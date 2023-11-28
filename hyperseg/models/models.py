#!/usr/bin/env python

from hyperseg.models import UNet, AGUNet, SpecTr

def replace_if_exists(paramname, defaultval, argdict):
    return argdict[paramname] if (paramname in argdict) else defaultval

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
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            mdmc_average=cfg.mdmc_average,
            bilinear=cfg.bilinear,
            batch_norm=cfg.batch_norm,
            class_weighting=cfg.class_weighting,
            export_preds_every_n_epochs=cfg.export_preds_every_n_epochs,
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
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            mdmc_average=cfg.mdmc_average,
            bilinear=cfg.bilinear,
            batch_norm=cfg.batch_norm,
            class_weighting=cfg.class_weighting,
            export_preds_every_n_epochs=cfg.export_preds_every_n_epochs,
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
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
            mdmc_average=cfg.mdmc_average,
            class_weighting=cfg.class_weighting,
            export_preds_every_n_epochs=cfg.export_preds_every_n_epochs,
            spatial_size=cfg.spatial_size,
            use_entmax15=cfg.use_entmax15,            
        )

    else:
        raise NotImplementedError(f"Model '{cfg.name}' does not exist.")
    
    return model
