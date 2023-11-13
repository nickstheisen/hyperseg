#!/usr/bin/env python

from hyperseg.models.imagebased import UNet, AGUNet, SpecTr

def replace_if_exists(paramname, defaultval, argdict):
    return argdict[paramname] if (paramname in argdict) else defaultval

def get_model(
        model_name,
        n_channels,
        n_classes,
        ignore_index,
        label_def,
        **kwargs
    ):
    if model_name == 'unet':
        model = UNet(
            n_channels=n_channels,
            n_classes=n_classes,
            ignore_index=ignore_index,
            label_def=label_def,
            loss_name=replace_if_exists("loss_name", "cross_entropy", kwargs),
            learning_rate=replace_if_exists("learning_rate", 0.001, kwargs),
            optimizer_name=replace_if_exists("optimizer_name", "AdamW", kwargs),
            momentum=replace_if_exists("momentum", 0.0, kwargs),
            weight_decay=replace_if_exists("weight_decay", 0.0, kwargs),
            mdmc_average=replace_if_exists("mdmc_average", "samplewise", kwargs),
            bilinear=replace_if_exists("bilinear", True, kwargs),
            batch_norm=replace_if_exists("batch_norm", True, kwargs),
            class_weighting=replace_if_exists("class_weighting", None, kwargs),
            export_preds_every_n_epochs=replace_if_exists("export_preds_every_n_epochs", 
                None, 
                kwargs)
        )
    elif model_name == 'agunet':
        model = AGUNet(
            n_channels=n_channels,
            n_classes=n_classes,
            label_def=label_def,
            ignore_index=ignore_index,
            loss_name=replace_if_exists("loss_name", "cross_entropy", kwargs),
            learning_rate=replace_if_exists("learning_rate", 0.001, kwargs),
            optimizer_name=replace_if_exists("optimizer_name", "AdamW", kwargs),
            momentum=replace_if_exists("momentum", 0.0, kwargs),
            weight_decay=replace_if_exists("weight_decay", 0.0, kwargs),
            mdmc_average=replace_if_exists("mdmc_average", "samplewise", kwargs),
            bilinear=replace_if_exists("bilinear", True, kwargs),
            batch_norm=replace_if_exists("batch_norm", True, kwargs),
            class_weighting=replace_if_exists("class_weighting", None, kwargs),
            export_preds_every_n_epochs=replace_if_exists("export_preds_every_n_epochs", 
                None, 
                kwargs)
        )
    elif model_name == 'spectr':
        model = SpecTr(
            n_channels=n_channels,
            n_classes=n_classes,
            label_def=label_def,
            ignore_index=ignore_index,
            loss_name=replace_if_exists("loss_name", "cross_entropy", kwargs),
            learning_rate=replace_if_exists("learning_rate", 0.001, kwargs),
            optimizer_name=replace_if_exists("optimizer_name", "AdamW", kwargs),
            momentum=replace_if_exists("momentum", 0.0, kwargs),
            weight_decay=replace_if_exists("weight_decay", 0.0, kwargs),
            mdmc_average=replace_if_exists("mdmc_average", "samplewise", kwargs),
            class_weighting=replace_if_exists("class_weighting", None, kwargs),
            export_preds_every_n_epochs=replace_if_exists("export_preds_every_n_epochs", 
                None, 
                kwargs),
            spatial_size=replace_if_exists("spatial_size", (1400,1800), kwargs),
            use_entmax15=replace_if_exists("use_entmax15", "entmax_bisect", kwargs),
            
        )

    elif model_name == 'ss3fcn':
        model = SS3FCN(
            n_channels=n_channels,
            n_classes=n_classes,
            label_def=label_def,
            ignore_index=ignore_index,
            loss_name=replace_if_exists("loss_name", "cross_entropy", kwargs),
            learning_rate=replace_if_exists("learning_rate", 0.001, kwargs),
            optimizer_name=replace_if_exists("optimizer_name", "AdamW", kwargs),
            momentum=replace_if_exists("momentum", 0.0, kwargs),
            weight_decay=replace_if_exists("weight_decay", 0.0, kwargs),
            mdmc_average=replace_if_exists("mdmc_average", "samplewise", kwargs),
            class_weighting=replace_if_exists("class_weighting", None, kwargs),
            export_preds_every_n_epochs=replace_if_exists("export_preds_every_n_epochs", 
                None, 
                kwargs)
        )
    else:
        raise NotImplementedError(f"Model '{model_name}' does not exist.")
    
    return model

