#!/usr/bin/env python

import hyperseg

from .hsidrive import HSIDrive
from .whuohs import WHUOHS
from .hyko2 import HyKo2
from .hyperspectralcity import HyperspectralCityV2
from .hsiroad import HSIRoad
from pathlib import Path

label_def_dir = Path(hyperseg.__file__).parent.joinpath("datasets/labeldefs")

def replace_if_exists(paramname, defaultval, argdict):
    return argdict[paramname] if (paramname in argdict) else defaultval

def get_datamodule(cfg):
    if cfg.name == 'hsidrive':
        datamodule = HSIDrive(
            basepath = cfg.basepath,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            train_prop=cfg.train_prop,
            val_prop=cfg.val_prop,
            manual_seed=cfg.manual_seed,
            precalc_histograms=cfg.precalc_histograms,
            normalize=cfg.normalize,
            label_def=label_def_dir.joinpath(cfg.label_def),
            spectral_average=cfg.spectral_average,
            ignore_water=cfg.ignore_water,      # water class is ignored 
            prep_3dconv=cfg.prep_3dconv,
            debug=cfg.debug,
        )
    elif cfg.name == 'whuohs':
        datamodule = WHUOHS(
            basepath = cfg.basepath,
            batch_size=cfg.batch_size,
            num_workers=cfg.num_workers,
            normalize=cfg.normalize,
            label_def=label_def_dir.joinpath(cfg.label_def),
            spectral_average=cfg.spectral_average,
            prep_3dconv=cfg.prep_3dconv,
            debug=cfg.debug,
        )
    elif dataset_name == 'hyko2':
        datamodule = HyKo2(
            filepath=basepath,
            num_workers=replace_if_exists("num_workers", 8, kwargs),
            batch_size=replace_if_exists("batch_size", 16, kwargs),
            label_set=replace_if_exists("label_set", "semantic", kwargs),
            train_prop=replace_if_exists("train_prop", 0.5, kwargs),
            val_prop=replace_if_exists("val_prop", 0.2, kwargs),
            manual_seed=replace_if_exists("manual_seed", 42, kwargs),
            label_def=replace_if_exists("label_def", 
                label_def_dir.joinpath("hyko2-semantic_labeldef.txt"),
                kwargs),
            spectral_average=replace_if_exists("spectral_average", True, kwargs),
            prep_3dconv=replace_if_exists("prep_3dconv", False, kwargs),
            debug=replace_if_exists("debug", False, kwargs),
        )
    elif dataset_name == 'hsiroad':
        datamodule = HSIRoad(
            basepath=basepath,
            sensortype=replace_if_exists("sensortype", "nir", kwargs),
            num_workers=replace_if_exists("num_workers", 8, kwargs),
            batch_size=replace_if_exists("batch_size", 32, kwargs),
            spectral_average=replace_if_exists("spectral_average", True, kwargs),
            label_def=replace_if_exists("label_def", 
                label_def_dir.joinpath("hsi-road_labeldef.txt"),
                kwargs),
            manual_seed=replace_if_exists("manual_seed", 42, kwargs),
            prep_3dconv=replace_if_exists("prep_3dconv", False, kwargs),
            debug=replace_if_exists("debug", False, kwargs),
        )
    elif dataset_name == 'hcv2':
        datamodule = HyperspectralCityV2(
            filepath=basepath,
            num_workers=replace_if_exists("num_workers", 8, kwargs),
            batch_size=replace_if_exists("batch_size", 4, kwargs),
            train_prop=replace_if_exists("train_prop", 0.5, kwargs),
            val_prop=replace_if_exists("val_prop", 0.1, kwargs),
            manual_seed=replace_if_exists("manual_seed", 42, kwargs),
            half_precision=replace_if_exists("half_precision", True, kwargs),
            label_def=replace_if_exists("label_def", 
                label_def_dir.joinpath("HCv2_labeldef.txt"),
                kwargs),
            spectral_average=replace_if_exists("spectral_average", True, kwargs),
            prep_3dconv=replace_if_exists("prep_3dconv", False, kwargs),
            debug=replace_if_exists("debug", False, kwargs),
        )

    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' does not exist.")

    return datamodule

            
            
            
