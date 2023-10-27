#!/usr/bin/env python

import hyperseg
from hyperseg.datasets.groundbased import HSIDrive, WHUOHS, HyKo2, HSIRoad, HyperspectralCityV2
from pathlib import Path

label_def_dir = Path(hyperseg.__file__).parent.joinpath("datasets/labeldefs")

def replace_if_exists(paramname, defaultval, argdict):
    return argdict[paramname] if (paramname in argdict) else defaultval

def get_datamodule(
        dataset_name,
        basepath,
        **kwargs
    ):
    if dataset_name == 'hsidrive':
        datamodule = HSIDrive(
            basepath = basepath,
            num_workers=replace_if_exists("num_workers", 8, kwargs),
            batch_size=replace_if_exists("batch_size", 32, kwargs),
            train_prop=replace_if_exists("train_prop", 0.6, kwargs),
            val_prop=replace_if_exists("val_prop", 0.2, kwargs),
            manual_seed=replace_if_exists("manual_seed", 42, kwargs),
            precalc_histograms=replace_if_exists("precalc_histograms", False, kwargs),
            normalize=replace_if_exists("normalize", False, kwargs),
            label_def=replace_if_exists("label_def", 
                label_def_dir.joinpath("hsidrive_labeldef_noWater.txt"),
                kwargs),
            spectral_average=replace_if_exists("spectral_average", True, kwargs),
            ignore_water=True,      # water class is ignored 
        )
    elif dataset_name == 'whuohs':
        datamodule = WHUOHS(
            basepath = basepath,
            batch_size=replace_if_exists("batch_size", 16, kwargs),
            num_workers = replace_if_exists("num_workers", 8, kwargs),
            normalize=replace_if_exists("normalize", False, kwargs),
            label_def=replace_if_exists("label_def", 
                label_def_dir.joinpath("whuohs_labeldef.txt"),
                kwargs),
            spectral_average=replace_if_exists("spectral_average", True, kwargs),
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
        )

    else:
        raise NotImplementedError(f"Dataset '{dataset_name}' does not exist.")
    
    return datamodule

            
            
            
