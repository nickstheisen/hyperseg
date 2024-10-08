#!/usr/bin/env python

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from hyperseg.datasets.utils import load_label_def
from tqdm import tqdm

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

import cv2

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rc('font', size=8)

def load_label_defs(label_def):
    label_defs = np.loadtxt(label_def, delimiter=',', dtype=str)
    label_names = np.array(label_defs[:,1])
    label_colors = np.array(label_defs[:,2:], dtype='int')
    return label_names, label_colors

class SpectrumPlotter():
    def __init__(
            self,
            dataset: Dataset,
            dataset_name: str,
            num_classes: int,
            class_def: str  ):
        self.dataloader = DataLoader(dataset,
                                batch_size=1,
                                shuffle=False,
                                num_workers=2,
                                persistent_workers=True)
        self.num_classes = num_classes
        self.dataset_name = dataset_name
        self.label_names, self.label_colors = load_label_def(class_def)
        self.label_colors = self.label_colors / 255.
        self.class_spectra = None
        self.class_spectra_np = None
        self.y_max = None
        self.y_min = None

    def _prepare_for_plotting(self, data):
        ## To efficiently plot the data we need to create one huge list containing all spectra 
        ## one after another. The spectra are seperated from each other by nan-values. Further, 
        ## DataShader uses pandas DataFrames so we need to convert the data to such.
        df = pd.DataFrame(data.T)
        
        ## append row with nan-values, these will be our seperators later
        df = pd.concat([df, 
            pd.DataFrame(
            [np.array([np.nan]*len(df.columns))], columns=df.columns, index=[np.nan]
            )]
        )
        x, y = df.shape

        ## rearrange 2D-array to get a 1D-Array where each column from the original array 
        ## is written after another.
        arr = df.values.reshape((x * y, 1), order='F')

        ## convert this list back to DataFrame as a column with header 'y' and use the x-values as
        ## index values
        df_r = pd.DataFrame(arr, columns=list('y'), index=np.tile(df.index.values, y)) 
        ## by resetting index, the current index value are converted to a new column (0) which
        ## will be used as x-values for plotting
        df_r = df_r.reset_index()
        df_r.columns.values[0] = 'x'
        return df_r

    def extract_class_samples(self):
        if self.class_spectra is not None:
            print("Already extracted class samples. Returning...")
            return
        
        # temporary dictionary to store class samples of each image in
        spectra_dict = dict()
        spectra_dict_np = dict()
        for c in range(self.num_classes):
            spectra_dict[c] = []
            spectra_dict_np[c] = []
        
        # iterate over dataset and extract class spectra
        print("#### Start Extraction ####")
        print(f"Nr. of classes: {self.num_classes}")
        for data, labels in tqdm(self.dataloader):
        
        # Below code kept for debugging
        #data, labels = next(iter(self.dataloader))
        #if True:

            # calculate y-limits
            if self.y_max is None or data.max() > self.y_max:
                self.y_max = data.max()
            if self.y_min is None or data.min() < self.y_min:
                self.y_min = data.min()
            
            # convert to correct shape
            self.n_channels = data.shape[1]
            data = np.squeeze(data).reshape(self.n_channels, -1).swapaxes(0,1)
            labels = np.squeeze(labels).reshape(-1)
            
            for c in range(self.num_classes):
                class_spectra = data[labels == c]#.numpy()

                # prepare data for efficient plotting
                dataframe = self._prepare_for_plotting(class_spectra)
                spectra_dict[c].append(dataframe)
                spectra_dict_np[c].append(class_spectra)
        
        print("#### Extraction finished ####")
        self.y_min = float(self.y_min)
        self.y_max = float(self.y_max)
        print("#### Aggregating Data ####")
        # store class spectra for further processing and analysis
        self.class_spectra = dict()
        self.class_spectra_np = dict()
        
        for c in range(self.num_classes):
            if len(spectra_dict[c]) == 0:
                self.class_spectra[c] = None
                print(f"Warning: No samples of class '{self.label_names[c]}' with ID '{c}' found.")
                continue
            self.class_spectra[c] = pd.concat(spectra_dict[c])
            self.class_spectra_np[c] = np.concatenate(spectra_dict_np[c])
        print("#### Aggregation Finished ####")
            
    def plot_color(self, out_dir, filetype='jpg',ylim=[0.0,2.0]):
        out_path = Path(out_dir)
        
        # iterate over dataset
        for data, labels in tqdm(self.dataloader):
            self.n_channels = data.shape[1]
            data = np.squeeze(data).reshape(self.n_channels, -1).swapaxes(0,1)
            labels = np.squeeze(labels).reshape(-1)
            
            # plot spectra
            for c in range(self.num_classes):
                plt.figure(c)
                class_spectra = data[labels == c]
                plt.plot(class_spectra.T, color=self.label_colors[c], alpha=0.02, linewidth=0.4)
        
        # export plots
        for c in range(self.num_classes):
            plt.figure(c)
            plt.title(self.label_names[c])
            plt.gca().set_ylim(ylim)
            plt.savefig(out_path.joinpath(f"{self.dataset_name}_class{c:02d}.{filetype}"))

    """
    based on: https://stackoverflow.com/questions/47175398/line-based-heatmap-or-2d-line-histogram
    """
    def plot_heatmap(self, out_dir, filetype='jpg'): 
        import datashader as ds
        import datashader.transfer_functions as tr
        import colorcet

        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if self.class_spectra is None:
            print("Error: Spectra must be extracted before plotting. Please call "
                   " function `extract_class_samples()`.")

        print(f"Nr. of channels: {self.n_channels}")
        print(f"#### Start Plotting ####")
        for c in range(self.num_classes):
            print(f"Class-ID: {c}")
            print(f"\tNr. Samples: {self.class_spectra_np[c].shape}\n")
            filename = out_path.joinpath(f"{self.dataset_name}_class{c:02d}.{filetype}")
            df = self.class_spectra[c]
            if df is None:
                continue

            # plotting params
            x_range = (df['x'].min(), df['x'].max())
            y_range = (self.y_min, self.y_max)
            ## binning granularity
            w = 1500
            h = 1000
            dpi = 150
            cvs = ds.Canvas(x_range=x_range, y_range=y_range, plot_height=h, plot_width=w)
            
            # aggregate data
            ## use column x as x-values and y for y-vales in plotting then count how many times a
            ## line passed one position in the plot
            aggs = cvs.line(df, 'x', 'y', ds.count())

            ## plot data with one color
            heatmap = tr.Image(tr.shade(aggs, cmap=colorcet.fire, how='linear'))

            # export data
            fig = plt.figure(c)
            ax = fig.add_subplot(111)
            ax.imshow(heatmap.to_pil())
            plt.title(self.label_names[c])

            # configure plot
            xstep = w/(self.n_channels-1)
            xticks = np.arange(0, w+xstep, xstep)
            ax.set_xticks(xticks)
            ax.set_xticklabels(np.arange(0,self.n_channels))
            plt.xticks(rotation = 45)
            # split y-axis into 5 sections
            ystep = h/4
            yticks = np.arange(0, h+ystep, ystep)
            ax.set_yticks(yticks)
            ax.set_yticklabels(
                np.round(np.arange(self.y_max, self.y_min, -(self.y_max-self.y_min)/5), 
                decimals=2)
            )

            # add marginal distribution to plot
            #ax = fig.add_subplot(224)
            #ax.plot(aggs.sum(axis=0)) # distribution along x-axis

            plt.savefig(filename, bbox_inches='tight', dpi=dpi)
            plt.clf()

class StatCalculator():
    
    def __init__(
                self,
                dataset: Dataset,
                batch_size=1,
                num_workers=2):

        self.dataset = dataset

        self.dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            persistent_workers=True)

    """
    Calculates mean and variance in a single pass using a batch-version of Welford's algorithm,
    described here: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
    """
    def getDatasetStats(self):
        data,_ = next(iter(self.dataloader))
        n_ch = data.shape[1]
        count = 0

        mean = torch.zeros(n_ch)
        std_dev = torch.zeros(n_ch)
        
        for data, _ in tqdm(self.dataloader):
            b, c, h, w = data.shape
            num_pixels = b * h * w
            mean += torch.sum(data, [0,2,3])
            count += num_pixels
        mean /= count
        for data, _ in tqdm(self.dataloader):
            centered = data - mean[None,:,None,None]
            sq_centered = centered **2
            std_dev += torch.sum(sq_centered, [0,2,3])

        std_dev /= count
        std_dev = std_dev ** 0.5
        return mean, std_dev
        print(f"__Calculated___\n{mean}\n {std_dev}")

        data, _ = next(iter(self.dataloader))
        n_ch = data.shape[1]
        
        count = 0
        mean = torch.zeros(n_ch)
        m2 = torch.zeros(n_ch)
    
        for data, _ in tqdm(self.dataloader):
            b, c, h, w = data.shape
            num_pixels = b * h * w
            count += num_pixels
            delta = data - mean[None,:,None, None]
            mean += torch.sum(delta, dim=[0,2,3]) / count
            delta2 = data - mean[None,:,None, None]
            m2 += torch.sum(torch.mul(delta, delta2), dim=[0,2,3])
        print(f"__Approximated__\n{mean}\n {m2/count}")
        return mean, m2/count

class ClassDistributionExtractor():

    def __init__(
                self,
                dataset: Dataset,
                label_def,
                batch_size=1,
                num_workers=2):

        self.dataset = dataset
        self.label_names, self.label_colors = load_label_defs(label_def)

        self.dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            persistent_workers=True)
        self.n_classes = len(self.label_names)
        self.aggregator = None     

    def extract(self):
        self.aggregator = np.zeros((self.n_classes), dtype=np.int64)
        for _, labels in tqdm(self.dataloader):
            for c in range(self.n_classes):
                self.aggregator[c] += np.count_nonzero(labels==c)
        return self.aggregator
    
    def set_abs_class_dist(self, class_dist):
        self.aggregator = class_dist

    def plot_bars(self, outpath, relative=False, plot_undef='False'):
        if self.aggregator is None:
            raise RuntimeError("Before plotting you must either extract the class distribution"
                " by calling `extract(..)` function or you need to provide the distribution"
                " through the function `set_abs_class_dist(..)` as numpy-array.")
        undef_idx = -1
        for i, name in enumerate(self.label_names):
            if 'undef' in name:
                undef_idx = i
                break

        class_dist = self.aggregator
        label_names = self.label_names
        label_colors = self.label_colors / 255.

        if undef_idx >= 0:
            class_dist = np.delete(class_dist, undef_idx)
            label_names = np.delete(label_names, undef_idx)
            label_colors = np.delete(label_colors, undef_idx, axis=0)
        
        if relative:
            class_dist = class_dist/class_dist.sum()
            class_dist *=100

        y_label = "Relative Abundance of class samples [%]" if relative else "Num. class samples"
        fig, ax = plt.subplots(layout='constrained')
        
        ax.bar(label_names, class_dist, color=label_colors)
        plt.xticks(label_names, rotation='vertical')
        ax.set_ylabel(y_label)
        
        plt.savefig(outpath)
        
class DatasetImageExporter():
    def __init__(
                self,
                dataset: Dataset,
                label_colors,
    ):

        self.dataset = dataset
        self.samplelist = dataset.samplelist()
        self.label_colors = label_colors

    def export(self, output_dir):
        output_dir_imgs = Path(output_dir).joinpath('imgs')
        output_dir_labels = Path(output_dir).joinpath('labels')
        output_dir_imgs.mkdir(parents=True, exist_ok=True)
        output_dir_labels.mkdir(parents=True, exist_ok=True)

        for i, samplename in enumerate(tqdm(self.samplelist)):
            image, label = self.dataset[i]

            # prepare image
            image = torch.mean(image, axis=0)
            # rescale image
            image = (image - image.min())/(image.max() - image.min()) 
            image *= 255

            # colorize labelmap
            label_img = self.label_colors[label]
            label_img = label_img[...,::-1]
            
            file_name = Path(Path(samplename).name).with_suffix('.jpg')
            image_outpath = output_dir_imgs.joinpath(file_name)
            cv2.imwrite(str(image_outpath), image.numpy())

            label_outpath = output_dir_labels.joinpath(file_name)
            cv2.imwrite(str(label_outpath), label_img)

