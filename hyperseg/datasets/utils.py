#!/usr/bin/env python

from tqdm import tqdm
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
import h5py
import sys

N_DEBUG_SAMPLES = 5

class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b   : int, optional
            Number of blocks transferred so far [default: 1].
        bsize   : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize   : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def load_label_def(label_def):
    label_defs = np.loadtxt(label_def, delimiter=',', dtype=str)
    label_names = np.array(label_defs[:,1])
    label_colors = np.array(label_defs[:, 2:], dtype='int')
    return label_names, label_colors

def label_histogram(dataset, n_classes):
    label_hist = torch.zeros(n_classes) # do not count 'unefined'(highest class_id)
    for i, (_, labels) in enumerate(DataLoader(dataset)):
        label_ids, counts = labels.unique(return_counts=True)
        for i in range(len(label_ids)):
            label_id = label_ids[i]
            if not (label_id == n_classes):
                label_hist[label_id] += counts[i]
    return label_hist

def apply_pca(n_components, origin_path, target_path, debug=False, half_precision=False):
    origin_path = Path(origin_path)
    target_path = Path(target_path)
    if target_path.exists():
        print(f"Target data set `{target_path}` already exists. Skipping PCA ...")
        return
    if not origin_path.exists():
        print(f"Origin data set `{origin_path}` does not exist. Exiting ...")
        sys.exit()
    print("Transforming data set with PCA ...")
    pca = PCA(n_components)
    with h5py.File(target_path, "w") as target_file, h5py.File(origin_path, "r") as origin_file:
        num_data = N_DEBUG_SAMPLES if debug else len(origin_file.keys()) 
        keys = list(origin_file.keys())[:N_DEBUG_SAMPLES] if debug else list(origin_file.keys())
        with tqdm(total=num_data) as pbar:
            for key in keys:
                group = target_file.create_group(key)

                # dim red. with pca
                data = np.array(origin_file[key]['image'])
                in_shape = data.shape
                out_shape = list(in_shape)
                out_shape[-1] = n_components
                X = data.reshape((-1, in_shape[-1]))
                Xt = pca.fit_transform(X)
                #print(f"pca variance ratio:{pca.explained_variance_ratio_}")
                transformed = Xt.reshape(out_shape)
                
                transformed_data = np.float16(transformed) if half_precision else np.float32(transformed)
                # write to file
                group.create_dataset("labels",  data=origin_file[key]['labels'])
                group.create_dataset("image", data=transformed_data)
                
                # update progress bar
                pbar.update(1)
