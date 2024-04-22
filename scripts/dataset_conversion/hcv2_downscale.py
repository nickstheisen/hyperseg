from hyperseg.datasets.pbdl_utils import hsd2npy
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import time
import h5py
import numpy as np
import cv2

parser = ArgumentParser(description="")
parser.add_argument("origin_dir", help="input directory.")
parser.add_argument("target_dir", help="output directory")

def rescale_dataset(origin_dir, target_dir, scaling_factor=0.5, mode='train'):
    origin_path = origin_dir.joinpath(f'hcv2_{mode}.h5')
    target_path = target_dir.joinpath(f'hcv2_rescaled{scaling_factor}_{mode}.h5')

    print(f"Start conversion of {mode} images!")
    print(f"Writing to {target_path}")

    t1 = time.time()
    with (h5py.File(target_path, "w") as target_file, 
            h5py.File(origin_path, "r") as origin_file):
        samplelist = list(origin_file.keys())

        for sample_name in tqdm(samplelist):
            if sample_name in target_file:
                continue
            labels = np.array(origin_file[sample_name]['labels'])
            image = np.array(origin_file[sample_name]['image'])
            labels = cv2.resize(labels, dsize=None, fx=scaling_factor, fy=scaling_factor, 
                                interpolation=cv2.INTER_NEAREST)
            image = cv2.resize(image, dsize=None, fx=scaling_factor, fy=scaling_factor, 
                                interpolation=cv2.INTER_NEAREST)
            group = target_file.create_group(sample_name)
            group.create_dataset("image", data=image)
            group.create_dataset("labels", data=labels)

    t2 = time.time()
    print(f"Converted {len(samplelist)} images in {t2-t1}s!")
    print()

if __name__ == '__main__':
    args = parser.parse_args()
    origin_dir = Path(args.origin_dir).absolute()
    target_dir = Path(args.target_dir).absolute()
    target_dir.mkdir(parents=True, exist_ok=True)

    rescale_dataset(origin_dir, target_dir, scaling_factor=0.5, mode='train')
    rescale_dataset(origin_dir, target_dir, scaling_factor=0.5, mode='test')
