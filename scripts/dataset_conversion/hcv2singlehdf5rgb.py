from hyperseg.datasets.pbdl_utils import hsd2npy
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import time
import h5py
import numpy as np
from imageio import imread
import cv2


parser = ArgumentParser(description="")
parser.add_argument("inputdir", help="input directory.")
parser.add_argument("outdir", help="output directory")

def convert_dir(basedir, outdir, mode='train'):
    data_dir = basedir.joinpath(mode)
    imgs = list(data_dir.glob('*_gray.png'))
    outfile = outdir.joinpath(f'hcv2_{mode}.h5')
    imgs = [str(img.stem).replace('_gray','') for img in imgs]

    scaling_factor=0.5

    print(f"Found {len(imgs)} in directory `{data_dir}`.")
    print()
    print(f"Start conversion of {mode} images!")
    print(f"Writing to {outfile}")
    t1 = time.time()
    with h5py.File(outfile, "a") as hdf_file:
        for imgstem in tqdm(imgs):
            if imgstem in hdf_file:
                continue
            labelpath = data_dir.joinpath(f'{imgstem}_gray.png')
            imgpath = data_dir.joinpath(f'{imgstem}.jpg')
            data = imread(imgpath)
            labels = imread(labelpath)
            
            # downscale
            labels = cv2.resize(labels, dsize=None, fx=scaling_factor, fy=scaling_factor, 
                                interpolation=cv2.INTER_NEAREST)
            data = cv2.resize(data, dsize=None, fx=scaling_factor, fy=scaling_factor, 
                                interpolation=cv2.INTER_NEAREST)
           
            if (data is not None and labels is not None):
                group = hdf_file.create_group(imgpath.stem)
                group.create_dataset("image", data=data)
                group.create_dataset("labels", data=labels)
            else:
                print(f"{imgpath} or {labelpath} is missing or empty.")

    t2 = time.time()
    print(f"Converted {len(imgs)} in {t2-t1}s!")
    print()
   

if __name__ == '__main__':
    args = parser.parse_args()
    inputdir = Path(args.inputdir).absolute()
    outdir = Path(args.outdir).absolute()
    outdir = outdir.joinpath('hcv2rgb')
    outdir.mkdir(parents=True, exist_ok=True)

    convert_dir(inputdir, outdir, 'train')
    convert_dir(inputdir, outdir, 'test')
