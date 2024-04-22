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
parser.add_argument("--show_only", action='store_true')

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

def conv2rgb(origin_dir, target_dir, dataset, bands, mode='train', show_only=True, minmax=None, transform_params=None):
    origin_path = origin_dir.joinpath(f'{dataset}_{mode}.h5')
    target_path = target_dir.joinpath(f'{dataset}_{mode}.h5')

    with (h5py.File(target_path, "w") as target_file, 
            h5py.File(origin_path, "r") as origin_file):
        samplelist = list(origin_file.keys())

        print(f"Start conversion of {len(samplelist)} image in {mode} set ...")
        t1 = time.time()
        i = 0
        for sample_name in tqdm(samplelist):
            if sample_name in target_file:
                continue
            if show_only:
                image = np.array(origin_file[sample_name]['image'])
                image = normalized_rgb(image, bands, 
                            minmax=minmax, 
                            transform_params=transform_params)
                cv2.imwrite(f'/home/hyperseg/rgb/{mode}/{i}.jpg', image[:,:,::-1])
                i +=1
                if i == 10:
                    break

            else:
                group = target_file.create_group(sample_name)
                labels = np.array(origin_file[sample_name]['labels'])
                group.create_dataset("labels", data=labels)

                image = np.array(origin_file[sample_name]['image'])
                image = normalized_rgb(image, bands, minmax=minmax)
                group.create_dataset("image", data=image)
               
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

def calc_rgb_mapping(origin_dir, target_dir, dataset, bands, modes, 
        acc_table_file=None, append_mean_stdev=False):
    if acc_table_file is None:
        acc_table = np.zeros((3,4,256)) 
        acc_table[:, 2, :] = 100
        acc_table[:, 3, :] = -100
    else:
        acc_table = np.load(acc_table_file)

    if append_mean_stdev:
        acc_table = np.pad(acc_table, ((0,0), (0,2), (0,0)), 'constant', constant_values=0)
        for b in bands:
            for i in range(acc_table.shape[2]):
                acc_table[b,4,i] = acc_table[b,0,i] / (acc_table[b,1,i]+1e-12) # mean
    # (rgb, (sum, count, min, max), (0..255)
    for mode in modes:
        origin_path = origin_dir.joinpath(f'{dataset}_{mode}.h5')
        target_path = target_dir.joinpath(f'{dataset}_{mode}.h5')

        with (h5py.File(target_path, "r") as target_file, 
                h5py.File(origin_path, "r") as origin_file):
            samplelist = list(origin_file.keys())
    
            for sample_name in tqdm(samplelist):
                target_image = np.array(target_file[sample_name]['image']).astype(np.uint8)
                origin_image = np.array(origin_file[sample_name]['image'])
            
                for b in bands:
                    target_image_b = target_image[:,:,b]
                    origin_image_b = origin_image[:,:,b]
                    for i in range(acc_table.shape[2]):
                        indicator_img = target_image_b == i
                        filtered_img = origin_image_b[indicator_img]
                        if filtered_img.size == 0:
                            img_min, img_max =  (100,-100) 
                        else:
                            img_min, img_max = filtered_img.min(), filtered_img.max()

                        if append_mean_stdev:
                            filtered_img -= acc_table[b,4,i] # - mean
                            acc_table[b,5,i] += np.square(filtered_img).sum() # square and accumulate
                        else: 
                            acc_table[b,0,i] += filtered_img.sum() # sum
                            acc_table[b,1,i] += indicator_img.sum() # count
                            if img_min < acc_table[b,2,i]: # min
                                acc_table[b,2,i] = img_min
                            if img_max > acc_table[b,3,i]: #max
                                acc_table[b,3,i] = img_max
    if append_mean_stdev:
        for b in bands:
            for i in range(acc_table.shape[2]):
                acc_table[b,5,i] /= (acc_table[b,1,i] + 1e-12) # divide by sample count
                acc_table[b,5,i] = np.sqrt(acc_table[b,5,i])
        outpath='/home/hyperseg/hcv_rgb_mapping_stats.npy'
    else:
        outpath='/home/hyperseg/hcv_rgb_mapping.npy'
    np.save(outpath, acc_table)
        
if __name__ == '__main__':
    DATASET_NAME = 'hyko2'
    args = parser.parse_args()
    origin_dir = Path(args.origin_dir).absolute()
    target_dir = Path(args.target_dir).absolute()
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
        conv2rgb(origin_dir, target_dir, args.dataset, bands, mode=mode, show_only=args.show_only, 
            minmax=minmax, transform_params=transform_params)
