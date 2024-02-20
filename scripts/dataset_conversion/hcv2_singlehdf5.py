from hyperseg.datasets.pbdl_utils import hsd2npy
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import time
import h5py
import numpy as np
from imageio import imread

parser = ArgumentParser(description="")
parser.add_argument("inputdir", help="input directory.")
parser.add_argument("outdir", help="output directory")

def convert_dir(basedir, outdir, mode='train'):
    image_dir = basedir.joinpath(mode).joinpath('images')
    label_dir = basedir.joinpath(mode).joinpath('labels')
    imgs = list(image_dir.glob('*.npy'))
    outfile = outdir.joinpath(f'hcv2_{mode}.h5')

    print(f"Found {len(imgs)} in directory `{image_dir}`.")
    print()
    print(f"Start conversion of {mode} images!")
    print(f"Writing to {outfile}")

    t1 = time.time()
    with h5py.File(outfile, "a") as hdf_file:
        for imgpath in tqdm(imgs):
            if imgpath.stem in hdf_file:
                continue
            labelpath = label_dir.joinpath(f'rgb{imgpath.stem}_gray.png')
            data = np.load(imgpath)
            labels = imread(labelpath)
            
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
    outdir = outdir.joinpath('hcv2')
    outdir.mkdir(parents=True, exist_ok=True)

    convert_dir(inputdir, outdir, 'train')
    convert_dir(inputdir, outdir, 'test')
