#!/usr/bin/env python3
import numpy as np
from math import ceil
import warnings
from pathlib import Path

from urllib.request import urlretrieve
import gdown
from tqdm import tqdm

from hyperseg.datasets.utils import TqdmUpTo

from scipy.io import loadmat
from skimage import io
import h5py

DATASETS_CONFIG = {
        'IP' : {
            'img': {
                'name': 'Indian_pines_corrected.mat',
                'url' : 'http://www.ehu.eus/ccwintco/uploads/6/67/Indian_pines_corrected.mat',
                'key' : 'indian_pines_corrected'
            },
            'gt': {
                'name' : 'Indian_pines_gt.mat',
                'url' : 'http://www.ehu.eus/ccwintco/uploads/c/c4/Indian_pines_gt.mat',
                'key' : 'indian_pines_gt'
            }
        },
        'PU' : {
            'img': {
                'name' : 'PaviaU.mat',
                'url'  : 'http://www.ehu.eus/ccwintco/uploads/e/ee/PaviaU.mat',
                'key'  : 'paviaU'
            },
            'gt': {
                'name' : 'PaviaU_gt',
                'url'  : 'http://www.ehu.eus/ccwintco/uploads/5/50/PaviaU_gt.mat',
                'key'  : 'paviaU_gt'
            }
        },
        'Salinas' : {
            'img': {
                'name': 'Salinas_corrected.mat',
                'url' : 'http://www.ehu.eus/ccwintco/uploads/a/a3/Salinas_corrected.mat',
                'key' : 'salinas_corrected'
            },
            'gt': {
                'name': 'Salinas_gt.mat',
                'url' : 'http://www.ehu.eus/ccwintco/uploads/f/fa/Salinas_gt.mat',
                'key' : 'salinas_gt'
            }
        },
        'SalinasA' : {
            'img': {
                'name': 'SalinasA_corrected.mat',
                'url' : 'http://www.ehu.eus/ccwintco/uploads/1/1a/SalinasA_corrected.mat',
                'key' : 'salinasA_corrected'
            },
            'gt': {
                'name': 'SalinasA_gt.mat',
                'url' : 'http://www.ehu.eus/ccwintco/uploads/a/aa/SalinasA_gt.mat',
                'key' : 'salinasA_gt'
            }
        },
        'PC' : {
            'img': {
                'name': 'Pavia.mat',
                'url' : 'http://www.ehu.eus/ccwintco/uploads/e/e3/Pavia.mat',
                'key' : 'pavia'
            },
            'gt': {
                'name': 'Pavia_gt.mat',
                'url' : 'http://www.ehu.eus/ccwintco/uploads/5/53/Pavia_gt.mat',
                'key' : 'pavia_gt'
            }
        },
        'KSC' : {
            'img': {
                'name': 'KSC.mat',
                'url' : 'http://www.ehu.es/ccwintco/uploads/2/26/KSC.mat',
                'key' : 'KSC'
            },
            'gt': {
                'name': 'KSC_gt.mat',
                'url' : 'http://www.ehu.es/ccwintco/uploads/a/a6/KSC_gt.mat',
                'key' : 'KSC_gt'
            }
        },
        'Botswana' : {
            'img': {
                'name': 'Botswana.mat',
                'url' : 'http://www.ehu.es/ccwintco/uploads/7/72/Botswana.mat',
                'key' : 'Botswana'
            },
            'gt': {
                'name': 'Botswana_gt.mat',
                'url' : 'http://www.ehu.es/ccwintco/uploads/5/58/Botswana_gt.mat',
                'key' : 'Botswana_gt'
            }
        },
        'AeroRIT' : {
            'img' : {
                'key' : 'data'
            },
            'gt' : {
                'key' : 'labels'
            },
            'reflectance' : {
                'img': {
                    'name': 'image_hsi_reflectance.tif',
                    'url' : 'https://drive.google.com/uc?id=1OC-g9RdMxtd2lzyrR5Wvd-ioSg-DLN-l'
                },
                'gt': {
                    'name': 'image_labels.tif',
                    'url' : 'https://drive.google.com/uc?id=1e7_x887BiCboCKGNYodm8czHoTHxT8ro'
                }
            },
            'radiance' : {
                'img': {
                    'name': 'image_hsi_radiance.tif',
                    'url' : 'https://drive.google.com/uc?id=15dL65NZqJSTRQaZzI9Ai33pLWWiAPnoZ'
                },
                'gt': {
                    'name': 'image_labels.tif',
                    'url' : 'https://drive.google.com/uc?id=1e7_x887BiCboCKGNYodm8czHoTHxT8ro'
                }
            },
            'full' : {
                'start_col': 0,
                'end_col': 3968
            },
            'left' : {
                'start_col': 0,
                'end_col': 1728
            },
            'mid' : {
                'start_col': 1728,
                'end_col': 2240
            },
            'right' : {
                'start_col': 2240,
                'end_col': 3968
            }
        }
}

AERORIT_COLOURLABELMAP= np.asarray(
                [[153, 0, 0], # << Differs from code in original repo s.t. unspecified gets ID 0
                [ 255, 0, 0],
                [ 0, 255, 0],
                [ 0, 0, 255],
                [ 0, 255, 255],
                [ 255, 127, 80]])

def list_datasets():
    dataset_list = ['IP', 'PU', 'Salinas', 'SalinasA', 'PC', 'KSC', 'Botswana', 'AeroRIT_radiance_full', 'AeroRIT_radiance_left', 'AeroRIT_radiance_mid', 'AeroRIT_radiance_right', 'AeroRIT_reflectance_full', 'AeroRIT_reflectance_left', 'AeroRIT_reflectance_mid', 'AeroRIT_reflectance_right']
    return dataset_list

def download_dataset(base_dir, scene):
    """
    Downloads hyperspectral dataset specified by `scene` and corresponding ground-truth labels into
    `base_dir` if it isn't already existing. Data and labels are then extracted and exported into
    a single hdf5-file with specified structure. The path to this file is returned.

    Parameters
    ----------
    base_dir : PosixPath, str
        Base-directory where dataset should be stored in.
    scene : str
        String defining a dataset or part of dataset (see DATASET_CONFIGS).

    Returns
    -------
    PosixPath : Path to hdf5-file in which dataset is stored.
    """

    # setup depends on dataset
    if len(scene.split('_')) == 1: # default
        modality, part = None, None # declaration necessary for instatiation check
        base_dir = Path(base_dir).expanduser().joinpath(scene)
        filepath_data = base_dir.joinpath(DATASETS_CONFIG[scene]['img']['name'])
        filepath_labels = base_dir.joinpath(DATASETS_CONFIG[scene]['gt']['name'])
        
    elif len(scene.split('_')) == 3: # AeroRIT
        scene, modality, part = scene.split('_')
        base_dir = Path(base_dir).expanduser().joinpath(scene)
        filepath_data = base_dir.joinpath(DATASETS_CONFIG[scene][modality]['img']['name'])
        filepath_labels = base_dir.joinpath(DATASETS_CONFIG[scene][modality]['gt']['name'])
    else :
        raise RuntimeError('Given scene unknown!')

    base_dir.mkdir(parents=True, exist_ok=True)

    # download data and load from file
    if filepath_data.suffix == '.mat': # datasets from ehu.es
        if not filepath_data.is_file():
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                    desc="Downloading {}".format(filepath_data)) as t:
                url = DATASETS_CONFIG[scene]['img']['url']
                urlretrieve(url, filename=filepath_data, reporthook=t.update_to)

        if not filepath_labels.is_file():
            with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                    desc="Downloading {}".format(filepath_labels)) as t:
                url = DATASETS_CONFIG[scene]['gt']['url']
                urlretrieve(url, filename=filepath_labels, reporthook=t.update_to)
        
        data = loadmat(filepath_data)[DATASETS_CONFIG[scene]['img']['key']]
        labels = loadmat(filepath_labels)[DATASETS_CONFIG[scene]['gt']['key']]

    elif filepath_data.suffix == '.tif': # aerorit
        if not filepath_data.is_file(): # download image if necessary
            print("Downloading {}".format(filepath_data))
            url = DATASETS_CONFIG[scene][modality]['img']['url']
            gdown.download(url=url, output=str(filepath_data), quiet=False)

        if not filepath_labels.is_file(): # download labels if necessary
            print("Downloading {}".format(filepath_labels))
            url = DATASETS_CONFIG[scene][modality]['gt']['url']
            gdown.download(url=url, output=str(filepath_labels), quiet=False)
        
        # extract part of image as defined in Rangnekar et al.
        base_dir = base_dir.joinpath(modality).joinpath(part)
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # check early if data exists already to avoid unecessarily loading and encoding data
        filepath_hdf = base_dir.joinpath(f'aerorit_{modality}_{part}.h5')
        if filepath_hdf.is_file():
            return filepath_hdf

        # extract defined part of dataset
        start_col = DATASETS_CONFIG[scene][part]['start_col']
        end_col = DATASETS_CONFIG[scene][part]['end_col']
    
        data = np.transpose(io.imread(filepath_data), (1,2,0))[53:,7:,:]
        data = data[:, start_col:end_col, :]

        labels = encode_labelmap(io.imread(filepath_labels), AERORIT_COLOURLABELMAP)[53:,7:]
        labels = labels[:, start_col:end_col]
        filepath_data = filepath_hdf

    filepath_hdf = filepath_data.with_suffix('.h5')
    
    # export data and labels to hdf
    if not filepath_hdf.is_file():
        with h5py.File(filepath_hdf, "w") as f:
            f.create_dataset("data", data=data)
            f.create_dataset("labels", data=labels)
            f.attrs['scene'] = scene
            if not modality is None:
                f.attrs['modality'] = modality
            if not part is None:
                f.attrs['part'] = part
        return filepath_hdf

    return filepath_hdf

def encode_labelmap(colour_img, colourlabelmap):
    """
    Takes a colour image, where each colour represents one specific label-class and replaces
    the colour value with label ids. The mapping is defined by array `colourlabelmap`.
    """
    colour_img = colour_img.astype(int)
    labels = np.zeros((colour_img.shape[0], colour_img.shape[1]), dtype=np.int16)
    for label_id, colour in enumerate(colourlabelmap):
        labels[np.where(np.all(colour == colour_img, axis=-1))] = label_id

    return labels

def load_data(datapath, labelpath, scene):
    scene = scene.split('_')[0]
    if datapath.suffix == '.mat':
        data = loadmat(datapath)[DATASETS_CONFIG[scene]['img']['key']]
        labels = loadmat(labelpath)[DATASETS_CONFIG[scene]['gt']['key']]
    else :
        raise RuntimeError('Unknown filetype.')
    return data, labels

def _sample_patches(imgs, 
        labelimgs, 
        patch_size, 
        patchgroup, 
        padding_mode, 
        padding_values, 
        ignore_labels,
        startidx=0):
    """
    Samples patches from data and stores them as dataset in hdf5-file. If dataset is already
    existing additional data is appended to existing data. 

    Parameters
    ----------
        imgs: list
            List of imgs to be sampled.
        labelimgs: list
            List of labelmaps corresponding to `imgs`.
        patch_size: int
            Size of sampled patches.
        patchgroup: Group
            h5py.Group where patches can be stored in.
        padding_mode: str or function
            Behaviour of padding. See numpy.pad(...) for further info.
        padding_values: sequence or scalar
            Values used for constant padding. See numpy.pad(...) for further info.
        ignore_labels: list
            Contains all labels that should be ignored. Samples with those labels are not saved.
        startidx: int, optional
            Defines startposition for writing data to dataset.

    Returns
    -------
        int: Number of valid samplepatches.

    """
    samplelist = []
    
    # number of bands should be constant, therefore the dimensionality can be read from any 
    # sub img
    bands = imgs[0].shape[-1]

    # calculate remapping for labels when removing `ignore_labels`
    # flatten labelimgs and convert to numpy array to use np.unique function on it
    flattened_labelimgs = np.concatenate([labelimg.reshape(-1) for labelimg in labelimgs])
    max_label = np.unique(flattened_labelimgs).max()
    remaining_labels = np.setdiff1d(np.arange(max_label+1), ignore_labels)
    label_remap = np.full((max_label+1), -1)
    for i, val in enumerate(remaining_labels):
        label_remap[val] = i

    valid_sample_count = 0
    for labelimg in labelimgs:
        valid_sample_count += np.invert(np.isin(labelimg, ignore_labels)).sum()
    print(f'Extracting {valid_sample_count} valid samples...')
    
    if ('data' in patchgroup) and ('labels' in patchgroup):
        # resize existing dataset to append patches from test set
        patchgroup['data'].resize((patchgroup['data'].shape[0] + valid_sample_count), axis=0)
        patchgroup['labels'].resize((patchgroup['labels'].shape[0] + valid_sample_count), axis=0)
    else:
        patchgroup.create_dataset('data', (valid_sample_count, patch_size, patch_size, bands)
                , chunks=(1, patch_size, patch_size, bands)
                , maxshape=(None, patch_size, patch_size, bands)
                , dtype=imgs[0].dtype) # datatype should be the same for all imgs
        patchgroup.create_dataset('labels', (valid_sample_count,1)
                , chunks=True, maxshape=(None, 1)
                , dtype=labelimgs[0].dtype) # datatype should be the same for all labelimgs
    
    idx = startidx
    with tqdm(total=valid_sample_count) as pbar:
        for img, labelimg in zip(imgs, labelimgs):

            # pad along spatial axes
            margin = int((patch_size - 1) / 2)
            X = np.pad(img, ((margin, margin), (margin, margin), (0,0)), 
                    mode=padding_mode, constant_values=padding_values) 

            # split patches
            for r in range(margin, X.shape[0] - margin):
                for c in range(margin, X.shape[1] - margin):
                    patchlabel = labelimg[r-margin, c-margin]

                    # do not create a sample for 'ignore_labels'
                    if patchlabel in ignore_labels:
                        continue
                    else :
                        # correct label
                        patchlabel = label_remap[patchlabel]

                    patch = X[r - margin:r + margin + 1, c - margin:c + margin + 1]
                    # store sample in hdf file
                    patchgroup['data'][idx] = patch
                    patchgroup['labels'][idx] = patchlabel

                    # update
                    idx += 1
                    pbar.update(1)

        patchgroup.attrs['patch_size'] = patch_size
        patchgroup.attrs['padding_mode'] = padding_mode
        patchgroup.attrs['padding_values'] = padding_values
        patchgroup.attrs['ignore_labels'] = ignore_labels

        return valid_sample_count

def split_random_sampling(inpath,
        patch_size, 
        train_prop,
        val_prop,
        outpath,
        padding_mode='constant', 
        padding_values=0, 
        ignore_labels=[0]):
        """
        Samples data patches at each position and assigns them to test and train set.
        
        The hyperspectral data cube is padded along its spatial dimensions. Then
        a sample patch is extracted at each position. Patches and corresponding labels are stored
        in hdf5-file. Also two lists are stored as datasets in hdf5-file that contain the indices
        of train and testpatches in patch-dataset.

        If sampled data already exists it is loaded. To resample please delete or move the existing
        file.

        Parameters
        ----------

        inpath : PosixPath, str
            Path to hdf5-file containing the data set
        patch_size: int
            Size of sampled patches
        train_prop: float
            Proportion of training data from all data.
        val_prop: float
            Proportion of validation data from all data.
        outpath: PosixPath, str
            Path to hdf5-file where sampled data is exported to
        padding_mode: str or function
            Behaviour of padding. See numpy.pad(...) for further info.
        padding_values: sequence or scalar
            Values used for constant padding. See numpy.pad(...) for further info.
        ignore_labels: list
            Contains all labels that should be ignored. Samples with those labels are not saved.
        startidx: int, optional
            Defines startposition for writing data to dataset.

        Returns
        -------
        PosixPath : Path to hdf5-file with data samples.

        """
        
        outdir = outpath.joinpath('random_sampling')
        outpath = outdir.joinpath(f'{patch_size}x{patch_size}_{padding_mode}_{train_prop:.2f}_{val_prop:.2f}.h5')
        
        if outpath.is_file():
            warnings.warn('Sampled data already exist, remove directory'
                ' "{}" to resample data!'.format(outpath))
            return outpath

        outdir.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(inpath, 'r') as in_file, h5py.File(outpath, 'w') as out_file:
            out_file.attrs.update(in_file.attrs) # copy attributes
            patchgroup = out_file.create_group('patches')
            
            # sample patches
            samplecount = _sample_patches([in_file['data'][:]], [in_file['labels'][:]], 
                        patch_size, 
                        patchgroup, 
                        padding_mode, 
                        padding_values, 
                        ignore_labels)        

            # define test, train and validation split
            split_idx_train = int(train_prop * samplecount)
            split_idx_val = int((train_prop + val_prop) * samplecount)

            samples = np.arange(samplecount)
            np.random.shuffle(samples)
            train_samples = samples[:split_idx_train]
            val_samples = samples[split_idx_train:split_idx_val]
            test_samples = samples[split_idx_val:]

            out_file.create_dataset('trainsample_list', data=train_samples)
            out_file.create_dataset('valsample_list', data=val_samples)
            out_file.create_dataset('testsample_list', data=test_samples)
            out_file.attrs['train_prop'] = train_prop
            out_file.attrs['val_prop'] = val_prop
            
        return outpath

def split_valid_sampling(inpath,
        patch_size, 
        train_prop,
        val_prop,
        outpath,
        padding_mode='constant', 
        padding_values=0, 
        ignore_labels=[0]):
        """
        Samples data patch at each position without overlap between test and train data.
        
        The hyperspectral data cube is divided into non-overlapping subimages. These subimages
        are divided into train and test data as specified by `train_prop`. Each subimage is
        then padded and sampled at each position to yield a patch for each position. The assignment 
        of nonoverlapping subimages into train and test set before random sampling avoids data 
        leakage.
        
        Patches and corresponding labels are stored
        in hdf5-file. Also two lists are stored as datasets in hdf5-file that contain the indices
        of train and testpatches in patch-dataset.

        If sampled data already exists it is loaded. To resample please delete or move the existing
        file.

        Parameters
        ----------

        inpath : PosixPath, str
            Path to hdf5-file containing the data set
        patch_size: int
            Size of sampled patches
        train_prop: float
            Proportion of training data from all data.
        val_prop: float
            Proportion of validation data from all data
        outpath: PosixPath, str
            Path to hdf5-file where sampled data is exported to
        padding_mode: str or function
            Behaviour of padding. See numpy.pad(...) for further info.
        padding_values: sequence or scalar
            Values used for constant padding. See numpy.pad(...) for further info.
        ignore_labels: list
            Contains all labels that should be ignored. Samples with those labels are not saved.
        startidx: int, optional
            Defines startposition for writing data to dataset.

        Returns
        -------
        PosixPath : Path to hdf5-file with data samples.

        """
        outdir = outpath.joinpath('valid_sampling')
        outpath = outdir.joinpath(f'{patch_size}x{patch_size}_{padding_mode}_{train_prop:.2f}_{val_prop:.2f}.h5')
        
        if outpath.is_file():
            warnings.warn('Sampled data already exist, remove directory'
                ' "{}" to resample data!'.format(outpath))
            return outpath

        outdir.mkdir(parents=True, exist_ok=True)
        
        with h5py.File(inpath, 'r') as in_file, h5py.File(outpath, 'w') as out_file:
            out_file.attrs.update(in_file.attrs) # copy attributes
            patchgroup = out_file.create_group('patches')

            data, labels = in_file['data'], in_file['labels']


            # split image into subimages of size patch_size x patch_size
            h, w, _ = data.shape
            num_subimg_h = ceil(h/patch_size) # patches along vertical axis
            num_subimg_w = ceil(w/patch_size) # patches along horizontal axis

            subimgs = []
            subimg_labels = []

            for i in range(num_subimg_h):
                for j in range(num_subimg_w):
                    start_idx_h = i*patch_size
                    start_idx_w = j*patch_size
                    end_idx_h = (i+1)*patch_size
                    end_idx_w = (j+1)*patch_size

                    # end_idx_h and end_idx_w may be greater than height and width of data array
                    if end_idx_h > h:
                        end_idx_h = h
                    if end_idx_w > w:
                        end_idx_w = w

                    subimgs.append(data[start_idx_h:end_idx_h, start_idx_w:end_idx_w])
                    subimg_labels.append(labels[start_idx_h:end_idx_h, start_idx_w:end_idx_w])

            # shuffle samples
            samples = list(zip(subimgs, subimg_labels))
            np.random.shuffle(samples)
            subimgs, subimg_labels = zip(*samples)

            # count how many pixels have non 'ignore_labels' and use result to assign approximately
            # train_prop share of non zero data to train set, val_prop of non zero data to validation set
            # and (1-(train_prop+val_prop)) to test set.
            if ignore_labels:
                cum_nonzero_labels = np.cumsum(
                        [np.invert(np.isin(lbls, ignore_labels)).sum() for lbls in subimg_labels])
                split_idx_train = 0
                split_idx_val = 0
                if cum_nonzero_labels[-1] == 0:
                    raise RuntimeError('Labelimage only contains ignored labels.')
                while(True):
                    if (cum_nonzero_labels[split_idx_train]/cum_nonzero_labels[-1]) < train_prop:
                        split_idx_train += 1
                    if (cum_nonzero_labels[split_idx_val]/cum_nonzero_labels[-1]) < (train_prop + val_prop):
                        split_idx_val += 1
                    else:
                        break
                print(f'{cum_nonzero_labels[split_idx_train]} / {cum_nonzero_labels[-1]}')
                print(f'{cum_nonzero_labels[split_idx_val]} / {cum_nonzero_labels[-1]}')
            else :
                split_idx_train = int(len(subimgs)*train_prop)
                split_idx_val = int(len(subimgs)*(train_prop + val_prop))

            # sample test and training data patches
            train_subimgs = subimgs[:split_idx_train]
            train_subimg_labels = subimg_labels[:split_idx_train]
            val_subimgs = subimgs[split_idx_train:split_idx_val]
            val_subimg_labels = subimg_labels[split_idx_train:split_idx_val]
            test_subimgs = subimgs[split_idx_val:]
            test_subimg_labels = subimg_labels[split_idx_val:]
            train_samplecount = _sample_patches(train_subimgs, train_subimg_labels, 
                        patch_size, 
                        patchgroup, 
                        padding_mode, 
                        padding_values, 
                        ignore_labels)
            val_samplecount = _sample_patches(val_subimgs, val_subimg_labels,
                        patch_size,
                        patchgroup,
                        padding_mode,
                        padding_values,
                        ignore_labels,
                        startidx=train_samplecount)
            test_samplecount = _sample_patches(test_subimgs, test_subimg_labels, 
                        patch_size, 
                        patchgroup, 
                        padding_mode, 
                        padding_values, 
                        ignore_labels,
                        startidx=(train_samplecount+val_samplecount))

            train_samples = np.arange(train_samplecount)
            val_samples = np.arange(train_samplecount, train_samplecount+val_samplecount)
            test_samples = np.arange((train_samplecount+val_samplecount), 
                    (train_samplecount+val_samplecount+test_samplecount))

            out_file.create_dataset('trainsample_list', data=train_samples)
            out_file.create_dataset('valsample_list', data=val_samples)
            out_file.create_dataset('testsample_list', data=test_samples)
            out_file.attrs['train_prop'] = train_prop
            out_file.attrs['val_prop'] = val_prop

        return outpath
