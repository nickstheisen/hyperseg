from pytorch_lightning import seed_everything, Trainer
import torch
from torch.cuda.amp import autocast
import hyperseg

import numpy as np
from pathlib import Path
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import cv2

from hyperseg.datasets import get_datamodule
from hyperseg.models import get_model

modes = ['train', 'val', 'test']
device = 'cuda:0'

def load_label_def(label_def):
        label_defs = np.loadtxt(label_def, delimiter=',', dtype=str)
        label_names = np.array(label_defs[:,1])
        label_colors = np.array(label_defs[:, 2:], dtype='int')
        return label_names, label_colors


def plot_predictions(dataset, model, outdir, label_colors, label_names, undef_idx):
    outdir.mkdir(parents=True, exist_ok=True)
    
    for i, (img, labels) in enumerate(tqdm(dataset)):
        if i == 100:
            break
        # move data to gpu
        img = img.cuda()

        # generate prediction and plot
        with autocast():
            model.eval()
            pred = torch.argmax(model(img), dim=1)
        
        pred = pred.cpu()
        pred_wm = pred
        pred = label_colors[pred].squeeze()
        pred = pred[:,:,::-1]
        cv2.imwrite(str(outdir.joinpath(f'pred{i}.png')), pred)
        cv2.imwrite(str(outdir.joinpath(f'gt{i}.png')), label_colors[labels].squeeze()[:,:,::-1])

        pred_wm[labels == undef_idx] = undef_idx
        pred_wm = label_colors[pred_wm].squeeze()
        pred_wm = pred_wm[:,:,::-1]
        cv2.imwrite(str(outdir.joinpath(f'maskedpred{i}.png')), pred_wm)


@hydra.main(version_base=None, config_path="../../run/conf", config_name="plot_predictions")
def main(cfg):

    seed_everything(42, workers=True)
    
    # check input
    if cfg.dataset.pca is not None:
        pca_out_path = Path(cfg.dataset.pca_out_dir)
        
        if not pca_out_path.is_dir():
            raise RuntimeError("`pca_out_dir` must be a directory")
    
    if cfg.outdir is None:
        raise RuntimeError("`outdir` was not specified in config!")

    # assemble outdir paths
    base_outdir = Path(cfg.outdir)
    base_outdir = base_outdir.joinpath(cfg.dataset.log_name)
    base_outdir = base_outdir.joinpath(cfg.model.log_name)
    outdirs = [base_outdir.joinpath(mode) for mode in modes]
       
    # configure datamodule
    cfg.dataset.batch_size=1
    datamodule = get_datamodule(cfg.dataset)
    datamodule.setup()

    # configure model
    with open_dict(cfg):
        cfg.model.n_channels = datamodule.n_channels
        cfg.model.n_classes = datamodule.n_classes
        cfg.model.spatial_size = list(datamodule.img_shape)
        cfg.model.ignore_index = datamodule.undef_idx
        cfg.model.label_def = datamodule.label_def

    model = get_model(cfg.model)
    checkpoint = torch.load(cfg.model.ckpt)
    del checkpoint['state_dict']['criterion.weight']
    model.load_state_dict(checkpoint['state_dict'])
    model.cuda()

    datasets =  [ datamodule.train_dataloader(), 
                datamodule.val_dataloader(), 
                datamodule.test_dataloader()]

    # configure label visualization

    label_def_dir = Path(hyperseg.__file__).parent.joinpath("datasets/labeldefs")
    label_def = label_def_dir.joinpath(cfg.dataset.label_def)
    label_names, label_colors = load_label_def(label_def)

    print("Begin plotting ...")
    for i in range(2,3):
        print(f"... {modes[i]} data to {outdirs[i]} ...")
        plot_predictions(datasets[i], model, outdirs[i], label_colors, label_names,
            cfg.model.ignore_index)
        #plot_predictions_withmask(datasets[i], model, outdirs[i], label_colors, label_names, 
        #    cfg.model.ignore_index)

if __name__ == '__main__':
    main()
