#!/usr/bin/env python

from tqdm import tqdm
import numpy as np

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
