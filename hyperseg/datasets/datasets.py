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
            basepath=cfg.basepath,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            label_def=label_def_dir.joinpath(cfg.label_def),
            manual_seed=cfg.manual_seed,
            normalize=cfg.normalize,
            spectral_average=cfg.spectral_average,
            pca=cfg.pca,
            pca_out_dir=cfg.pca_out_dir,
            debug=cfg.debug,
            ignore_water=cfg.ignore_water,      # water class is ignored 
            drop_last=cfg.drop_last,
        )
    elif cfg.name == 'whuohs':
        datamodule = WHUOHS(
            basepath=cfg.basepath,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            label_def=label_def_dir.joinpath(cfg.label_def),
            manual_seed=cfg.manual_seed,
            normalize=cfg.normalize,
            spectral_average=cfg.spectral_average,
            pca=cfg.pca,
            pca_out_dir=cfg.pca_out_dir,
            debug=cfg.debug,
            drop_last=cfg.drop_last,
        )
    elif cfg.name == 'hyko2':
        datamodule = HyKo2(
            basepath=cfg.basepath,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            label_def=label_def_dir.joinpath(cfg.label_def),
            manual_seed=cfg.manual_seed,
            normalize=cfg.normalize,
            spectral_average=cfg.spectral_average,
            pca=cfg.pca,
            pca_out_dir=cfg.pca_out_dir,
            debug=cfg.debug,
            drop_last=cfg.drop_last
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
            debug=cfg.debug,
            drop_last=cfg.drop_last,
        )
    elif cfg.name == 'hcv2':
        datamodule = HyperspectralCityV2(
            basepath=cfg.basepath,
            num_workers=cfg.num_workers,
            batch_size=cfg.batch_size,
            label_def=label_def_dir.joinpath(cfg.label_def),
            manual_seed=cfg.manual_seed,
            normalize=cfg.normalize,
            spectral_average=cfg.spectral_average,
            pca=cfg.pca,
            pca_out_dir=cfg.pca_out_dir,
            debug=cfg.debug,
            half_precision=cfg.half_precision,
            drop_last=cfg.drop_last,
        )

    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' does not exist.")

    return datamodule

            
            
            
