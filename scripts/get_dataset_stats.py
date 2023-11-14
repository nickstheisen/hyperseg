#!/usr/bin/env python

from hyperseg.datasets import HSIRoadDataset
from hyperseg.datasets import HSDataset
from hyperseg.datasets import HSIDriveDataset
from hyperseg.datasets import WHUOHSDataset

from hyperseg.datasets.analysis.tools import StatCalculator
from hyperseg.datasets.prep import download_dataset

import torch
from torchvision import transforms
from hyperseg.datasets.transforms import ToTensor, PermuteData, ReplaceLabels

import numpy as np


'''
############# HSI-Road ############################################################################
transform_hsiroad = transforms.Compose([
    ToTensor(),
])


filepath_hsiroad = '/home/hyperseg/data/hsi_road/hsi_road'
hsiroad = HSIRoadDataset(
            data_dir=filepath_hsiroad, 
            transform=transform_hsiroad,
            mode='full',
            collection='nir')

statcalc_hsiroad = StatCalculator(hsiroad, batch_size=32, num_workers=8)
mean_hsiroad, var_hsiroad = statcalc_hsiroad.getDatasetStats()
np.savetxt('/mnt/data/dataset-stats/hsiroad-stats.txt', 
            torch.stack((mean_hsiroad, var_hsiroad)))

############# HyKo2-VIS ###########################################################################
transform_hyko = transforms.Compose([
    ToTensor(),
    PermuteData(new_order=[2,0,1]),
])

filepath_hyko = download_dataset('~/data', 'HyKo2-VIS_Semantic')
hyko = HSDataset(filepath_hyko, transform=transform_hyko)

statcalc_hyko = StatCalculator(hyko, batch_size=16, num_workers=8)
mean_hyko, var_hyko = statcalc_hyko.getDatasetStats()
np.savetxt('/mnt/data/dataset-stats/hykovis-stats.txt', 
            torch.stack((mean_hyko, var_hyko)))
############## HCV2-PCA1 ###########################################################################
transform_hcv2 = transforms.Compose([
    ToTensor(half_precision=True),
    PermuteData(new_order=[2,0,1]),
])

filepath_hcv2 = '/mnt/data/HyperspectralCityV2_PCA1.h5'
hcv2 = HSDataset(filepath_hcv2, transform=transform_hcv2)
statcalc_hcv2 = StatCalculator(hcv2, batch_size=4, num_workers=4)
mean_hcv2, var_hcv2 = statcalc_hcv2.getDatasetStats()
np.savetxt('/mnt/data/dataset-stats/hcv2-stats.txt', 
            torch.stack((mean_hcv2, var_hcv2)))
'''
############## HSI-Drive ##########################################################################
'''
transform_hsidrive = transforms.Compose([
    ToTensor(),
])

filepath_hsidrive = '/mnt/data/data/hsi-drive/Image_dataset'
hsidrive = HSIDriveDataset(filepath_hsidrive, transform=transform_hsidrive)
statcalc_hsidrive = StatCalculator(hsidrive, batch_size=32, num_workers=8)
mean_hsidrive, var_hsidrive = statcalc_hsidrive.getDatasetStats()
np.savetxt('/mnt/data/dataset-stats/hsidrive-stats.txt', 
            torch.stack((mean_hsidrive, var_hsidrive)))
'''
############## WHUOHS ##############################################################################
transform_whuohs = transforms.Compose([
                    ToTensor(),
                    PermuteData(new_order=[2,0,1]),
                    ReplaceLabels({0:24, 24:0})
                    ])

filepath_whuohs = "/home/hyperseg/datadisk/datasets/WHU-OHS"
whuohs = WHUOHSDataset(filepath_whuohs, transform_whuohs, mode='full')
statcalc_whuohs = StatCalculator(whuohs, batch_size=32, num_workers=8)
mean_whuohs, var_whuohs = statcalc_hsidrive.getDatasetStats()
np.savetxt('/mnt/data/dataset-stats/whuohs-stats.txt',
            torch.stack((mean_whuohs, var_whuohs)))

