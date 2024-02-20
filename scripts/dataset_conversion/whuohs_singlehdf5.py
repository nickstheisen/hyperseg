from hyperseg.datasets.pbdl_utils import hsd2npy
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import time
import h5py
import numpy as np
from imageio import imread
from scipy.io import loadmat
import tifffile

parser = ArgumentParser(description="")
parser.add_argument("inputdir", help="input directory.")
parser.add_argument("outdir", help="output directory")

def convert_dir(dataset_name, basedir, outdir, mode='train'):
    basedir = Path(basedir).joinpath(mode)
    img_dir = basedir.joinpath('image')
    label_dir = basedir.joinpath('label')
    imgs = list(img_dir.glob('*.tif'))
    outfile = outdir.joinpath(f'{dataset_name}_{mode}.h5')

    print(f"Found {len(imgs)} in directory `{img_dir}`.")
    print()
    print(f"Start conversion of {mode} images!")
    print(f"Writing to {outfile}")

    t1 = time.time()
    with h5py.File(outfile, "a") as hdf_file:
        for imgpath in tqdm(imgs):
            if imgpath.stem in hdf_file:
                continue
            data = tifffile.imread(imgpath).astype(np.float32) / 10000.0
            labels = tifffile.imread(label_dir.joinpath(imgpath.name))
            
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
    DATASET_NAME = 'whuohs'
    args = parser.parse_args()
    inputdir = Path(args.inputdir).absolute()
    outdir = Path(args.outdir).absolute()
    outdir = outdir.joinpath(DATASET_NAME)
    outdir.mkdir(parents=True, exist_ok=True)

    convert_dir(DATASET_NAME, inputdir, outdir, 'train')
    convert_dir(DATASET_NAME, inputdir, outdir, 'test')
    convert_dir(DATASET_NAME, inputdir, outdir, 'val')
