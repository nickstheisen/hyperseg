# Hyperseg

A framework for **hyperspectral semantic segmentation** based on pytorch and pytorch-lightning. 

### Supported Datasets
* [HyKo2](https://wp.uni-koblenz.de/hyko/) ([paper](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w3/html/Winkens_HyKo_A_Spectral_ICCV_2017_paper.html))
* [HSI-Drive v2.0](https://ipaccess.ehu.eus/HSI-Drive/) ([paper]([https://ipaccess.ehu.eus/HSI-Drive/files/IVS_2021_web.pdf](https://ieeexplore.ieee.org/document/9575298)))
* [Hyperspectral City V2.0](https://pbdl-ws.github.io/pbdl2021/challenge/download.html)
* [HSI-Road](https://github.com/NUST-Machine-Intelligence-Laboratory/hsi_road) ([paper](https://ieeexplore.ieee.org/document/9102890))
* [WHU-OHS](https://github.com/zjjerica/WHU-OHS-Pytorch) ([paper](https://www.sciencedirect.com/science/article/pii/S1569843222002102))

### Supported Models
* U-Net ([paper](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28))
* Attention-Gated U-Net ([paper](https://ieeexplore.ieee.org/document/9306920); not well-tested)
* SpecTr ([paper](https://arxiv.org/abs/2103.03604))

## Setup

### Installation
* install Nvidia driver + CUDA (tested: driver 470.82/525.125.06 with CUDA 11.4/12.0)
* create environment using python=3.11
* `pip install -e .`

### Dataset Preparation

1. **Hyperseg**: `python scripts/prep_hyko.py`

2. **HSI-Drive**: Can be downloaded [here](https://ipaccess.ehu.eus/HSI-Drive/). For uncompressing the zip-file a password is required. Follow the instructions on the website

3. **Hyperspectral City v2.0**: Can be downloaded [here](https://pbdl-ws.github.io/pbdl2021/challenge/download.html). To use the our DataLoader it is required to decompress the data and convert it to a hdf5-file (see [convert_hsds_to_hdf](https://github.com/nickstheisen/hyperseg/blob/main/hyperseg/datasets/pbdl_utils.py#L56). At the time of our experimentation only train set from the competition was available which we split up into train/test/validation. 

4. **HSI-Road**: Follow instructions on [project page](https://github.com/NUST-Machine-Intelligence-Laboratory/hsi_road)

5. **WHUOHS**: See [project page](https://github.com/zjjerica/WHU-OHS-Pytorch) or [zenodo page](https://zenodo.org/records/7258035#.ZCvESnZByUl).


### Usage


Configure dataset basepaths in dataset configuration files (`hyperseg/run/conf/dataset/<dataset>.yaml`) and logdir (`hyperseg/run/conf/<train/test_conf>.yaml`). 
We use hydra for config management. You can customize train and test runs by overwriting parameters (`field=value`) when running the script in terminal. Hydras multi-run feature allows you to automatically run multiple experiments with different parameterizations (see conf-files and [hydra docs](https://hydra.cc/docs/intro/) for more information).


**Training/Testing**

Per default a vanilla unet and HyKo2 dataset is used. Models as well as datasplits can be found [here](https://drive.google.com/drive/folders/1W55NqiP6Lb5SLxD1xYF8NKiF4fG89UyQ?usp=drive_link).


```bash
# train
python run/train.py logging.project_name=<project_name>
```

```bash
# test
python run/test.py logging.project_name=<project_name> model.ckpt=<path_to_checkpoint>
```

## Benchmark

|Dataset|Approach|$R_\mu$|$R_M$|$F_{1_{M}}$|$J_M$|
|---|---|---|---|---|---|
|HCV2|U-Net                              |85.25|48.62|48.18|37.73|
| |RU-Net (PCA1)                         |88.25|58.07|55.43|44.26|
|HyKo2-VIS|U-Net                        |85.36|68.15|68.55|57.39|
| |RU-Net (pRGB)                         |89.18|73.92|75.04|64.67|
|HSI-Drive-V2|U-Net                     |94.95|74.74|76.08|64.95|
| |RU-Net (PCA1)|                        |97.02|86.80|87.76|79.23|
| | |                                   |    |    |    |    |
|**Average </br> Performance**|U-Net    |88.52|63.84|64.27|53.36|
| |RU-Net (pRGB)                         |91.15|71.09|71.80|61.34|
|**Worst-Case </br> Performance**|U-Net |82.25|48.63|48.18|37.73|
| |RU-Net (PCA1)                        |85.61|58.07|55.43|44.26|
