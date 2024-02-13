from hyperseg.datasets.pbdl_utils import hsd2npy
from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm
import time

parser = ArgumentParser(
            description="Converts all .hsd-files in subdirectories `train/images` and `test/images` to .npy-files. New files are stored next to the original files.")
parser.add_argument("basedir", help="the base directory of HCV2 dataset.")
parser.add_argument("--half-precision", action='store_true',
        help="if set data is stored in 16bit (half-precision) to save memory")

if __name__ == '__main__':
    args = parser.parse_args()
    basedir = Path(args.basedir).absolute()
    train_img_dir = basedir.joinpath('train/images')
    test_img_dir = basedir.joinpath('test/images')
    half_precision = args.half_precision
    
    train_imgs = list(train_img_dir.glob('*.hsd'))
    test_imgs = list(test_img_dir.glob('*.hsd'))
    print(f"half-precision: {args.half_precision}")
    print(f"Found {len(train_imgs)} in directory `{train_img_dir}`.")
    print(f"Found {len(test_imgs)} in directory `{test_img_dir}`.")
    print()
    print(f"Start conversion of train images ...")
    t1 = time.time()
    for img in train_imgs:
        hsd2npy(img, img.with_suffix('.npy'), args.half_precision)
    t2 = time.time()
    print(f"Converted {len(train_imgs)} in {t2-t1}s!")
    print()
    print(f"Start conversion of test images ...")
    t1 = time.time()
    for img in test_imgs:
        hsd2npy(img, img.with_suffix('.npy'), args.half_precision)
    t2 = time.time()
    print(f"Converted {len(test_imgs)} in {t2-t1}s!")
    
