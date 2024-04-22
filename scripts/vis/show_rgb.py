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

def load_label_def(label_def):
        label_defs = np.loadtxt(label_def, delimiter=',', dtype=str)
        label_names = np.array(label_defs[:,1])
        label_colors = np.array(label_defs[:, 2:], dtype='int')
        return label_names, label_colors


def plot_dataset(dataset, label_colors, label_names):
    min_val = 100
    max_val = -100
    for i, (img, labels) in enumerate(tqdm(dataset)):
        img_min = img.min()
        img_max = img.max()
        if img_min < min_val:
            min_val = img_min
        if img_max > max_val:
            max_val = img_max
        
    print(min_val, max_val)
    '''
        if i == 20:
            continue
        img = img.squeeze()
        img = np.array([img[62,:,:], img[18,:,:], img[0,:,:]])
        #for i in range(3):
        #    tmp = img[i,:,:]
        #    img[i,:,:] = ((tmp - tmp.min())/(tmp.max() - tmp.min()))*255
        img = ((img - img.min())/(img.max() - img.min()))*255
        img = img.transpose(1,2,0)
        img = img[:,:,::-1]
        print(img.shape)
        print(dataset.dataset.samplelist()[0:5])
        #img = img.astype(np.int8)
        cv2.imwrite('/tmp/test.jpg', img)
        print(np.isnan(img).any())
        break
    '''
    '''
        # plot image as single channel grayscale
        single_ch_img = img[0, 0].numpy()
        single_ch_img = ((single_ch_img - single_ch_img.min())
                        /(single_ch_img.max() - single_ch_img.min()))*255
        cv2.imwrite(str(outdir.joinpath(f'img{i}.jpg')), single_ch_img)
        
        # plot label image
        label_img = label_colors[labels].squeeze()
        label_img = label_img[:,:,::-1]
        cv2.imwrite(str(outdir.joinpath(f'label{i}.jpg')), label_img)
    '''

@hydra.main(version_base=None, config_path="../../run/conf", config_name="plot_dataset")
def main(cfg):

    seed_everything(42, workers=True)
    
    # check input
    if cfg.dataset.pca is not None:
        pca_out_path = Path(cfg.dataset.pca_out_dir)
        
        if not pca_out_path.is_dir():
            raise RuntimeError("`pca_out_dir` must be a directory")
    
       
    # configure datamodule
    cfg.dataset.batch_size=1
    datamodule = get_datamodule(cfg.dataset)
    datamodule.setup()

    datasets =  [ #datamodule.train_dataloader()]#, 
                #datamodule.val_dataloader(), 
                datamodule.test_dataloader()]

    # configure label visualization

    label_def_dir = Path(hyperseg.__file__).parent.joinpath("datasets/labeldefs")
    label_def = label_def_dir.joinpath(cfg.dataset.label_def)
    label_names, label_colors = load_label_def(label_def)

    print("Begin plotting ...")
    for i in range(1):
        print(f"... {modes[i]} data to {i} ...")
        plot_dataset(datasets[i], label_colors, label_names)

if __name__ == '__main__':
    main()
