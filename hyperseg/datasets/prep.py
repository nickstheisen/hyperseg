#!/usr/bin/env python

import numpy as np
from tqdm import tqdm
import h5py
from pathlib import Path
from urllib.request import urlretrieve
from zipfile import ZipFile
from scipy.io import loadmat
import sys

from hyperseg.datasets.utils import TqdmUpTo

DATASETS_CONFIG = {
        'HyKo2-VIS': {
            'name': 'vis_annotated.zip',
            'url': 'https://hyko-proxy.uni-koblenz.de/hyko-dataset/HyKo2/vis/vis_annotated.zip',
            'Semantic': {
                'data_key' : 'data',
                'label_key' : 'label_Semantic Classes for Urban Scenes',
            },
            'Material': {
                'data_key' : 'data',
                'label_key': 'label_spectral_reflectances'
            }
        },
        'HyKo2-NIR': {
            'name': 'nir_annotated.zip',
            'url':'https://hyko-proxy.uni-koblenz.de/hyko-dataset/HyKo2/nir/nir_annotated.zip',
            'Semantic': {
                'data_key' : 'data',
                'label_key': 'label_Semantic Classes for Urban Scenes',
            },
            'Material': {
                'data_key' : 'data',
                'label_key': 'label_spectral_reflectances'
            }

        },
}

def download_dataset(base_dir, name):
    # configure paths for data dirs
    if len(name.split('_')) == 2: # HyKo2 Datasets
        name, label_semantics = name.split('_')
        base_dir = Path(base_dir).expanduser().joinpath(name)
        filepath_data = base_dir.joinpath(DATASETS_CONFIG[name]['name'])
    else : 
        raise RuntimeError(f'Dataset {name} unknown!')

    # create dir if not existing
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # download zip archive
    if not filepath_data.is_file():
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
            desc="Downloading {}".format(filepath_data)) as t:
            url = DATASETS_CONFIG[name]['url']
            urlretrieve(url, filename=filepath_data, reporthook=t.update_to)

    # export data into hdf5
    filepath_hdf = base_dir.joinpath(f'{name}_{label_semantics}.h5')

    ## extract data + labels from archive and store in hdf5-file
    with ZipFile(filepath_data, 'r') as zip_ref:
        if not filepath_hdf.is_file():
            with h5py.File(filepath_hdf, "w") as f:
                f.attrs['name'] = name
                f.attrs['label_semantics'] = label_semantics
                for filename in zip_ref.namelist():
                    mat = loadmat(zip_ref.open(filename))
                    data = mat.get(DATASETS_CONFIG[name][label_semantics]['data_key'])
                    labels = mat.get(DATASETS_CONFIG[name][label_semantics]['label_key'])
                
                    # skip if data or label do not exist
                    if data is not None and labels is not None :
                        matgroup = f.create_group(filename)
                        matgroup.create_dataset("data",
                            data=data)
                        matgroup.create_dataset("labels",
                            data=labels)
                    else :
                        # print name of file and keys to find broken or incomplete files
                        print(f'File {filename} is incomplete. Keys are {mat.keys()}.')
    return filepath_hdf 
