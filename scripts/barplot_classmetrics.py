#!/usr/bin/env python

from argparse import ArgumentParser
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.rc('font', size=8)

def plot_classmetrics(files, 
                relative=False, 
                outpath=None, 
                labelfile=None,
                y_label=None,
                legend_entries=None):

    width = 0.4 # width of bars
    multiplier = 0.0
    undef_idx = -1

    fig, ax = plt.subplots(layout='constrained')
    
    baseline = np.loadtxt(files[0])
    x = np.arange(len(baseline))
    
    if labelfile is not None:
        labels = np.loadtxt(labelfile, dtype=str, delimiter=',')[:,1]
        labels = [label.strip() for label in labels]
        if 'undefined' in labels:
            undef_idx = labels.index('undefined')
            del labels[undef_idx]
            x = np.delete(x, -1)
    
    min_val = 100
    max_val = -100
    if relative:
        del files[0]
        le_bl = legend_entries[0]
        del legend_entries[0]
    
    for idx, f in enumerate(files):
        if relative:
            data = (np.loadtxt(f) - baseline) * 100.
        else:
            data = np.loadtxt(f) * 100.

        if undef_idx >= 0:
            data = np.delete(data, undef_idx)

        if data.max() > max_val:
            max_val = data.max()
        if data.min() < min_val:
            min_val = data.min()

        offset = width * multiplier
        if legend_entries is None:
            rects = ax.bar(x + offset, data, width, label=idx)
        else:
            rects = ax.bar(x + offset, data, width, label=legend_entries[idx])
        ax.bar_label(rects, padding=3, rotation='vertical', fmt='%.2f')
        multiplier += 1

    ax.set_ylabel(y_label)
    ax.legend(loc='lower right', bbox_to_anchor=(0.65,1.0))
    ax.set_xticks(x + width)
    ax.set_ylim(min_val-5, max_val+25)

    if labelfile is not None:
        ax.set_xticklabels(labels, rotation=90)
    
    if outpath is None:
        plt.show()
    else:
        plt.savefig(outpath, dpi=fig.dpi*3)

if __name__ == '__main__':
    parser = ArgumentParser(
                prog='barplot_classmetrics.py',
                description='Creates barplot of class-metrics next to each other.',
            )

    parser.add_argument('-r', '--relative', action='store_true',
                help=('First file is used as baseline. Only the absolute difference to the baseline'
                    ' is plotted.'
                )
    )
    parser.add_argument('files', metavar='f', type=str, nargs='+',
                help=('Path to files with metric values for different runs. If `--relative` is '
                    'used the first file defines the baseline relative to which all other runs'
                    ' are plotted.')
    )
    parser.add_argument('-o', '--outfile', type=str,
                help=('Where should the plot be stored. If not set plot is only shown.')
    )
    parser.add_argument('-l', '--labels', type=str,
                help=('txt-file containing at least the class names')
    )
    parser.add_argument('-y', '--y_label', type=str,
                help=('label for y-axis')
    )
    parser.add_argument('-e', '--legend_entries', type=str, nargs='+',
                help=('Legend entries (must be same amount as files).')
    )

    args = parser.parse_args()
    plot_classmetrics(files=args.files, 
                    relative=args.relative, 
                    outpath=args.outfile, 
                    labelfile=args.labels,
                    y_label=args.y_label,
                    legend_entries=args.legend_entries)
