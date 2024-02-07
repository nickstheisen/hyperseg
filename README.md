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
* `pip install .`

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

|Dataset|Approach|$R_mu$|$R_M$|$F_{1_{M}}$|$J_M$|
|---|---|---|---|---|---|
|HCV|U-Net                              |89.2|52.6|54.2|45.6|
| |HS3-Baseline                         |93.1|60.0|62.7|54.3|
|HyKo2-VIS|U-Net                        |85.9|68.6|68.8|57.8|
| |HS3-Baseline                         |88.5|72.6|72.5|62.2|
|HSI-Drive-V2|U-Net                     |95.0|76.1|76.8|65.9|
| |HS3-Baseline|                        |96.0|76.7|82.7|72.7|
| | |                                   |    |    |    |    |
|**Average </br> Performance**|U-Net    |90.0|65.8|66.6|56.4|
| |HS3-Baseline                         |92.7|70.8|72.6|63.0|
|**Worst-Case </br> Performance**|U-Net |85.9|52.6|54.2|45.6|
| |HS3-Baseline                         |88.5|72.6|72.5|62.22
