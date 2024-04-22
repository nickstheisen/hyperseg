from pytorch_lightning import seed_everything
import numpy as np
from pathlib import Path
from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import cv2
import hyperseg

from hyperseg.datasets import get_datamodule

modes = ['train', 'val', 'test']

ctrs = None

def load_label_def(label_def):
        label_defs = np.loadtxt(label_def, delimiter=',', dtype=str)
        label_names = np.array(label_defs[:,1])
        label_colors = np.array(label_defs[:, 2:], dtype='int')
        return label_names, label_colors


def count_samples(dataset):
    for _, labels in tqdm(dataset):
        for i in range(ctrs.shape[0]):
            ctrs[i] += (labels == i).sum()

@hydra.main(version_base=None, config_path="../../run/conf", config_name="plot_dataset")
def main(cfg):
    global ctrs

    seed_everything(42, workers=True)
    
    # check input
    if cfg.dataset.pca is not None:
        pca_out_path = Path(cfg.dataset.pca_out_dir)
        
        if not pca_out_path.is_dir():
            raise RuntimeError("`pca_out_dir` must be a directory")
    
    if cfg.outdir is None:
        raise RuntimeError("`outdir` was not specified in config!")

    # assemble outdir paths
    outdir = Path(cfg.outdir)
       
    # configure datamodule
    cfg.dataset.batch_size=1
    datamodule = get_datamodule(cfg.dataset)
    datamodule.setup()

    datasets =  [ datamodule.train_dataloader(), 
                datamodule.val_dataloader(), 
                datamodule.test_dataloader()]

    # configure label visualization

    label_def_dir = Path(hyperseg.__file__).parent.joinpath("datasets/labeldefs")
    label_def = label_def_dir.joinpath(cfg.dataset.label_def)
    label_names, label_colors = load_label_def(label_def)
    ctrs = np.zeros(label_names.shape, dtype=np.longlong)

    print("Counting samples in  ...")
    for i in range(3):
        print(f"... {modes[i]} data ...")
        count_samples(datasets[i])

    outdir.mkdir(parents=True, exist_ok=True)
    outfile_abs = outdir.joinpath(f'{cfg.dataset.log_name}_dist_abs.txt')
    outfile_rel = outdir.joinpath(f'{cfg.dataset.log_name}_dist_rel.txt')
    np.savetxt(outfile_abs, ctrs)
    np.savetxt(outfile_rel, ctrs/ctrs.sum())
    
if __name__ == '__main__':
    main()
