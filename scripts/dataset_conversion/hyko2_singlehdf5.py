from hyperseg.datasets.pbdl_utils import hsd2npy
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import time
import h5py
import numpy as np
from imageio import imread
from scipy.io import loadmat

parser = ArgumentParser(description="")
parser.add_argument("inputdir", help="input directory.")
parser.add_argument("outdir", help="output directory")

def convert_dir(dataset_name, basedir, outdir, mode='train'):
    file_dir = basedir.joinpath(mode)
    imgs = list(file_dir.glob('*.mat'))
    outfile = outdir.joinpath(f'{dataset_name}_{mode}.h5')

    print(f"Found {len(imgs)} in directory `{file_dir}`.")
    print()
    print(f"Start conversion of {mode} images!")
    print(f"Writing to {outfile}")

    t1 = time.time()
    with h5py.File(outfile, "a") as hdf_file:
        for imgpath in tqdm(imgs):
            if imgpath.stem in hdf_file:
                continue
            mat = loadmat(imgpath)
            data = mat.get('data')
            labels = mat.get('label_Semantic Classes for Urban Scenes')
            
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
    DATASET_NAME = 'hyko2'
    args = parser.parse_args()
    inputdir = Path(args.inputdir).absolute()
    outdir = Path(args.outdir).absolute()
    outdir = outdir.joinpath(DATASET_NAME)
    outdir.mkdir(parents=True, exist_ok=True)

    convert_dir(DATASET_NAME, inputdir, outdir, 'train')
    convert_dir(DATASET_NAME, inputdir, outdir, 'test')
    convert_dir(DATASET_NAME, inputdir, outdir, 'val')
