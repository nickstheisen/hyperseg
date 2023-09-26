import torch
from sklearn.decomposition import PCA
import numpy as np
from torchvision import transforms

class ToTensor(object):
    """ Convert tuples of hyperspectral data and labels to tensors."""
    def __init__(self, half_precision=False):
        self.half_precision=half_precision

    def __call__(self, sample):
        patch, label = sample
        patch = torch.from_numpy(patch)
        label = torch.from_numpy(label).type(torch.long)

        if self.half_precision:
            patch = patch.type(torch.float16)
        else:
            patch = patch.type(torch.float32)
        return (patch, label)

class InsertEmptyChannelDim(object):
    """ Insert Empty Channel dimension to apply 3D-Convolutions to hyperspectral images tensors."""

    def __call__(self, sample):
        patch, label = sample
        patch = torch.unsqueeze(patch, 0)

        return (patch, label)

class PermuteData(object):
    """ Permutes sample-data dimensions as defined in `new_order`. """
    def __init__(self, new_order):
        self.new_order = new_order

    def __call__(self, sample):
        patch, label = sample
        patch = patch.permute(self.new_order)

        return (patch, label)

class ReplaceLabel(object):
    """ Replace label `orig_lbl` with `new_lbl`."""
    def __init__(self, orig_lbl, new_lbl):
        self.orig_lbl = orig_lbl
        self.new_lbl = new_lbl

    def __call__(self, sample):
        patch, label = sample
        label[label == self.orig_lbl] = self.new_lbl

        return (patch, label)

class ReplaceLabels(object):
    """ Replace labels based on mapping given as dictionary `label_mapping`"""
    def __init__(self, label_mapping):
        self.label_mapping = label_mapping

    def __call__(self, sample):
        patch, label = sample
        res_label = torch.clone(label)
        for k, v in self.label_mapping.items() : res_label[label == k] = v

        return (patch, res_label)

class PCADR(object):
    """ Reduce dimensionality using PCA to `target_dims`. Parameters are fitted for each batch."""
    def __init__(self, target_dim):
        self.target_dim = target_dim

    def __call__(self, sample):
        in_patch, label = sample
        pca = PCA(n_components=self.target_dim)

        in_shape = in_patch.shape
        out_shape = list(in_shape)
        out_shape[-1] = self.target_dim

        X = np.reshape(in_patch, (-1, in_shape[-1]))
        X = pca.fit_transform(X)
        out_patch = np.reshape(X, out_shape)

        return (out_patch, label)

class Normalize(object):
    """Normalize hyperspectral image with given channel means `means` and channel stds `stds`."""
    def __init__(self, means, stds):
        self._norm = transforms.Normalize(mean=means, std=stds)
        self.means = means
        self.stds = stds

    def __call__(self, sample):
        patch, label = sample
        patch = self._norm(patch)
        #patch = patch - self.means[None,:,None,None]

        return (patch, label)
        
class SpectralAverage(object):
    """Reduce spectral dimension of images to 1 by averaging all channels."""
    def __init__(self):
        pass

    def __call__(self, sample):
        patch, label = sample
        patch = patch.mean(dim=0, keepdim=True)

        return (patch, label)
