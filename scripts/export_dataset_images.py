#!/usr/bin/env python

from hyperseg.datasets.analysis.tools import DatasetImageExporter, load_label_defs
from hyperseg.datasets.transforms import ToTensor, PermuteData, ReplaceLabels
from hyperseg.datasets.groundbased import WHUOHSDataset

from torchvision import transforms
from argparse import ArgumentParser
import numpy as np
from pathlib import Path

valid_datasets = ['whuohs']

def get_dataset(dataset_name, basepath, dataset_split):
    if dataset_name not in valid_datasets:
        raise RuntimeError(f"Dataset must be one of {valid_datasets}.")

    if dataset_name == 'whuohs':
        transform_whuohs = transforms.Compose([
                    ToTensor(),
                    PermuteData(new_order=[2,0,1]),
                    ReplaceLabels({0:24, 24:0})
                    ])

        dataset = WHUOHSDataset(basepath, transform_whuohs, mode=dataset_split)

    return dataset

if __name__ == '__main__':
    parser = ArgumentParser(
                prog='export_dataset_images.py',
                description='Export images in dataset as grayscale image to output folder.'
    )
    
    parser.add_argument('dataset_name', choices=valid_datasets,
            help="name of dataset from which class distribution should be extracted." )
    parser.add_argument('dataset_basepath', help="path to dataset folder")
    parser.add_argument('label_def', help="path to label definition file")
    parser.add_argument('dataset_split', choices=('train', 'val', 'test'),
            help="which split is exported.")
    parser.add_argument('out_dir', 
            help='Path to output directory.')

    args = parser.parse_args()
    
    outpath = Path(args.out_dir).joinpath(args.dataset_split)
    
    _, label_colors = load_label_defs(args.label_def)
    dataset = get_dataset(args.dataset_name, args.dataset_basepath, args.dataset_split)
    exporter = DatasetImageExporter(dataset, label_colors)
    
    exporter.export(outpath)
