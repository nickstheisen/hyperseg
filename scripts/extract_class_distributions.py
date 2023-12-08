#!/usr/bin/env python

from hyperseg.datasets.analysis.tools import ClassDistributionExtractor
from hyperseg.datasets.transforms import ToTensor, PermuteData, ReplaceLabels
from hyperseg.datasets import WHUOHSDataset

from torchvision import transforms
from argparse import ArgumentParser
import numpy as np
from pathlib import Path

valid_datasets = ['whuohs']

def get_dataset(dataset_name, basepath):
    if dataset_name not in valid_datasets:
        raise RuntimeError(f"Dataset must be one of {valid_datasets}.")

    if dataset_name == 'whuohs':
        transform_whuohs = transforms.Compose([
                    ToTensor(),
                    PermuteData(new_order=[2,0,1]),
                    ReplaceLabels({0:24, 24:0})
                    ])

        dataset = WHUOHSDataset(basepath, transform_whuohs, mode='full')

    return dataset

if __name__ == '__main__':
    parser = ArgumentParser(
                prog='extract_class_distribution.py',
                description='Extract (and plot) class distribution of dataset.'
    )
    
    parser.add_argument('dataset_name', choices=valid_datasets,
            help="name of dataset from which class distribution should be extracted." )
    parser.add_argument('dataset_basepath', help="path to dataset folder")
    parser.add_argument('label_def', help="path to label definition file")
    parser.add_argument('csv_out_file', 
            help=("path to csv-file where result should stored. If file already exists, the result"
            " is loaded instead of extracting the them again. To overwrite activate the switch"
            " `--o/--overwrite`."))
    parser.add_argument('-p', '--plot_out_file', 
            help="path to location where plot should be stored.")
    parser.add_argument('-o', '--overwrite', action='store_true',
            help="Overwrite csv-file from previous extraction if it already exists.")
    parser.add_argument('-r', '--relative', action='store_true',
            help=("When plotting defines if the class distribution is given in absolute label"
            " counts or relative in percent (default: absolute)."))

    args = parser.parse_args()
    
    dataset = get_dataset(args.dataset_name, args.dataset_basepath)
    cdex = ClassDistributionExtractor(dataset, args.label_def)

    csv_out_file = Path(args.csv_out_file)

    if csv_out_file.exists() and not args.overwrite:
        results = np.loadtxt(csv_out_file)
        cdex.set_abs_class_dist(results)
    else:
        results = cdex.extract()
        np.savetxt(csv_out_file, results)
    
    if args.plot_out_file is not None:
        cdex.plot_bars(args.plot_out_file, relative=args.relative)
    
    
