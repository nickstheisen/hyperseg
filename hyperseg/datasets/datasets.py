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
    elif cfg.name == 'hyko2':
        datamodule = HyKo2(
            filepath=cfg.basepath,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            label_set=cfg.label_set,
            train_prop=cfg.train_prop,
            val_prop=cfg.val_prop,
            manual_seed=cfg.manual_seed,
            label_def=label_def_dir.joinpath(cfg.label_def),
            spectral_average=cfg.spectral_average,
            prep_3dconv=cfg.prep_3dconv,
            debug=cfg.debug,
        )
    elif cfg.name == 'hsiroad':
        datamodule = HSIRoad(
            basepath=cfg.basepath,
            sensortype=cfg.sensortype,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            spectral_average=cfg.spectral_average,
            label_def=label_def_dir.joinpath(cfg.label_def),
            manual_seed=cfg.manual_seed,
            prep_3dconv=cfg.prep_3dconv,
            debug=cfg.debug,
        )
    elif cfg.name == 'hcv2':
        datamodule = HyperspectralCityV2(
            filepath=cfg.basepath,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            train_prop=cfg.train_prop,
            val_prop=cfg.val_prop,
            manual_seed=cfg.manual_seed,
            half_precision=cfg.half_precision,
            label_def=label_def_dir.joinpath(cfg.label_def),
            spectral_average=cfg.spectral_average,
            prep_3dconv=cfg.prep_3dconv,
            debug=cfg.debug,
        )

    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' does not exist.")

    return datamodule

            
            
            
