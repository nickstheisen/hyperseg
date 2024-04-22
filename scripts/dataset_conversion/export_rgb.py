from hyperseg.datasets.pbdl_utils import hsd2npy
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import time
import h5py
import numpy as np
from imageio import imread
from scipy.io import loadmat
import matplotlib.pyplot as plt
import cv2

datasets = ["hyko2", "hcv2", "hcv2rawrgb", "hsidrive"]

parser = ArgumentParser(description="")
parser.add_argument("origin_dir", help="input directory.")
parser.add_argument("target_dir", help="output directory")
parser.add_argument("dataset", help="name of dataset", choices=datasets)

def gamma_correct(img, gamma):
    lookUpTable = np.empty((1,256), np.uint8)
    for i in range(256):
        lookUpTable[0,i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(img, lookUpTable)
    return res

def get_band(hsi, band):
    if band == -1:
        return np.zeros_like(hsi[:,:,0])
    else:
        return hsi[:,:,band]
def normalized_rgb(hsi, bands, minmax=None, transform_params=None):
    rgb = np.array([get_band(hsi, b) for b in bands])
    #clahe = cv2.createCLAHE()
    # channel wise
    for i in range(3):
        if minmax is None:
            rgb_min, rgb_max = (rgb[i,:,:].min(), rgb[i,:,:].max())
        else:
            rgb_min, rgb_max = minmax[i]
        a, b, c = transform_params[i]
        #rgb[i,:,:] = (((rgb[i,:,:] - rgb_min)/(rgb_max - rgb_min))*255)
        #rgb[i,:,:] = clahe.apply(rgb_uint8[i,:,:])
        #rgb[i,:,:] = cv2.equalizeHist(rgb_uint8[i,:,:])
        #rgb[i,:,:] = gamma_correct(rgb_uint8[i,:,:], 0.3)
        rgb[i,:, :] = (inv_exp_func(rgb[i,:,:], a,b,c)*255).astype(np.uint8)

    rgb = rgb.transpose([1,2,0])
    return rgb

def inv_exp_func(x, a, b, c):
    print(a, b, c)
    return -(1./b)* np.log((x-c)/a)

def export_rgb(origin_dir, target_dir, dataset, mode='train'):
    origin_path = origin_dir.joinpath(f'{dataset}_{mode}.h5')

    with (h5py.File(origin_path, "r") as origin_file):
        samplelist = list(origin_file.keys())

        print(f"Start conversion of {len(samplelist)} image in {mode} set ...")
        t1 = time.time()
        i = 0
        for sample_name in tqdm(samplelist):
            image = np.array(origin_file[sample_name]['image'])
            image *= 1
            print(image.min())
            print(image.max())
            image = image.astype(np.uint8)
            cv2.imwrite(str(target_dir.joinpath(f'{i}.jpg')), image[:,:,:])
            i +=1
            if i == 20:
                break

        t2 = time.time()
        print(f"Done! Required Time (s) : {t2-t1}")

def global_band_min_max(origin_dir, dataset, bands, mode):
    origin_path = origin_dir.joinpath(f'{dataset}_{mode}.h5')
    with (h5py.File(origin_path, "r") as origin_file):
        samplelist = list(origin_file.keys())
        minmaxs = np.repeat([[100.,-100.]], 3, axis=0)
        for sample_name in tqdm(samplelist):
            if not ('image' in origin_file[sample_name].keys()):
                continue
            image = np.array(origin_file[sample_name]['image'])
            for i, b in enumerate(bands):
                min_b = image[:,:,b].min()
                max_b = image[:,:,b].max()
                if minmaxs[i,0] > min_b:
                    minmaxs[i,0] = min_b
                if minmaxs[i,1] < max_b:
                    minmaxs[i,1] = max_b
        return minmaxs

if __name__ == '__main__':
    DATASET_NAME = 'hyko2'
    args = parser.parse_args()
    origin_dir = Path(args.origin_dir).absolute()
    target_dir = Path(args.target_dir).absolute()
    target_dir = target_dir.joinpath(f"{args.dataset}_rgb")
    target_dir.mkdir(parents=True, exist_ok=True)

    bands = None
    minmax = None
    if args.dataset == 'hyko2':
        bands = [14,7,0] # rgb
        modes = ['train', 'test', 'val']
    elif args.dataset == 'hcv2':
        bands = [62,18,0] #rgb
        modes = ['train', 'test']
        minmax = [[-0.017,  0.52],
                  [-0.015,  0.83],
                  [-0.031,  1.13 ]]
    elif args.dataset == 'hcv2rawrgb':
        args.dataset = 'hcv2'
        bands = [0,1,2] #rgb
        modes = ['train', 'test']
        #minmax = [[-0.017,  0.52],
        #          [-0.015,  0.83],
        #          [-0.031,  1.13 ]]
        transform_params = [[ 1.02857371e-03, -4.76046254e+00,  1.44713289e-02],
 [ 1.88666792e-04, -6.71615679e+00,  1.84434871e-02],
 [ 1.72012654e-04, -5.92827886e+00,  9.85248706e-03]]
    elif args.dataset == 'hsidrive':
        bands = [5,2,0] #rgb
        modes = ['train','test', 'val']
        minmax = [[0,4.],
                  [0,4.],
                  [0,4.]]

    else:
       raise RuntimeError(f"{args.dataset} invalid!")

    #calc_rgb_mapping(origin_dir, target_dir, args.dataset, bands, modes=modes, 
    #                acc_table_file='/home/hyperseg/hcv_rgb_mapping.npy',
    #                append_mean_stdev=True)
    for mode in modes:
        #print(global_band_min_max(origin_dir, args.dataset, bands, mode))
        target_dir_mode = target_dir.joinpath(mode)
        target_dir_mode.mkdir(parents='True', exist_ok='True')
        export_rgb(origin_dir, target_dir_mode, args.dataset, mode=mode)
