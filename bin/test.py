#!/usr/bin/env python

from pathlib import Path

from hsdatasets.remotesensing.prep import download_dataset, split_random_sampling, split_secure_sampling
from hsdatasets.remotesensing import RemoteSensingDataset
from hsdatasets.transforms import ToTensor, InsertEmptyChannelDim, PermuteData

from hyperseg.models.patchbased import SSRN

import torch
from torchvision import transforms

if __name__ == '__main__':
    scene = 'AeroRIT_radiance_mid'

    # download data if not already existing
    dataset_path = download_dataset(base_dir='~/data', scene=scene)
    
    # sample data and split into test and trainset
    filepath = split_secure_sampling(dataset_path, 7, 0.7, dataset_path.parents[0])

    # create dataset
    transform = transforms.Compose([
        ToTensor(), 
        InsertEmptyChannelDim(),
        PermuteData(new_order=[0,3,1,2])
        ])
    salinasa_train = RemoteSensingDataset(filepath, train=True, apply_pca=False, pca_dim=10, transform=transform)
    # create dataloader
    trainloader = torch.utils.data.DataLoader(salinasa_train,
            batch_size=16, shuffle=True, num_workers=2)

    net = SSRN(num_bands=51, num_classes=5)

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        print(i, inputs.shape, labels.shape)
        net(inputs)
        if i == 1:
            break;

