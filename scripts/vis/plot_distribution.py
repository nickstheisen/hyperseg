from pytorch_lightning import seed_everything
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from argparse import ArgumentParser

import hyperseg
from hyperseg.datasets import get_datamodule

parser = ArgumentParser(
            description='Plots the label distribution of a dataset as bar plot.')
parser.add_argument('data_filepath')
parser.add_argument('labeldef_filepath')

def load_label_def(label_def):
        label_defs = np.loadtxt(label_def, delimiter=',', dtype=str)
        label_names = np.array(label_defs[:,1])
        label_colors = np.array(label_defs[:, 2:], dtype='int')
        return label_names, label_colors

def get_undef_idx(label_names):
    for i, name in enumerate(label_names):
        if 'undef' in name:
            return i

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i]+0.005, f'{y[i]:.4f}', ha='center', 
            bbox = dict(facecolor='white', alpha=.8))

def plot_label_distribution(args):
    label_names, label_colors = load_label_def(args.labeldef_filepath)
    data = np.loadtxt(args.data_filepath)

    # sort
    sort_order = np.argsort(data)[::-1]
    sorted_data = [data[i] for i in sort_order]
    sorted_names = [label_names[i] for i in sort_order]
    sorted_colors = [label_colors[i] for i in sort_order]

    undef_idx = get_undef_idx(sorted_names)
    del sorted_data[undef_idx]
    del sorted_names[undef_idx]
    del sorted_colors[undef_idx]

    sorted_colors = np.array(sorted_colors)
    sorted_colors = np.divide(sorted_colors, 255.)
    
    plt.bar(sorted_names, sorted_data, color=sorted_colors, edgecolor='black')
    addlabels(sorted_names, sorted_data)
    plt.xticks(rotation=30, ha='right')
    plt.show()

if __name__ == '__main__':
    args = parser.parse_args()
    plot_label_distribution(args)
