<p align="center"><img src="https://github.com/user-attachments/assets/e741234c-00f8-4d19-93e2-48f7ef1c27b3" alt="hyperseg logo" width=300/></p>


A framework for **hyperspectral semantic segmentation** based on pytorch and pytorch-lightning. This repository is part of a publication at 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2024) with the title 

*"HS3-Bench: A Benchmark and Strong Baseline for Hyperspectral Semantic Segmentation in Driving Scenarios"* 

Find the accepted version of the on arXiv: [paper](http://arxiv.org/abs/2409.11205)

### Supported Datasets
* [HyKo2](https://wp.uni-koblenz.de/hyko/) ([paper](https://openaccess.thecvf.com/content_ICCV_2017_workshops/w3/html/Winkens_HyKo_A_Spectral_ICCV_2017_paper.html))
* [HSI-Drive v2.0](https://ipaccess.ehu.eus/HSI-Drive/) ([paper]([https://ipaccess.ehu.eus/HSI-Drive/files/IVS_2021_web.pdf](https://ieeexplore.ieee.org/document/9575298)))
* [Hyperspectral City V2.0](https://pbdl-ws.github.io/pbdl2021/challenge/download.html)
* [HSI-Road](https://github.com/NUST-Machine-Intelligence-Laboratory/hsi_road) ([paper](https://ieeexplore.ieee.org/document/9102890))
* [WHU-OHS](https://github.com/zjjerica/WHU-OHS-Pytorch) ([paper](https://www.sciencedirect.com/science/article/pii/S1569843222002102))

### Supported Models
* U-Net ([paper](https://link.springer.com/chapter/10.1007/978-3-319-24574-4_28))
* DeeplabV3+ ([paper](https://link.springer.com/chapter/10.1007/978-3-030-01234-2_49))
* RU-Net (regularized U-Net, ours)

## Setup

### Installation
* install Nvidia driver + CUDA (tested: driver 470.82/525.125.06 with CUDA 11.4/12.0)
* create environment using python=3.11
* `pip install -e .`

### Dataset Preparation

1. **Hyperseg**: `python scripts/prep_hyko.py`

2. **HSI-Drive**: Can be downloaded [here](https://ipaccess.ehu.eus/HSI-Drive/). For uncompressing the zip-file a password is required. Follow the instructions on the website

3. **Hyperspectral City v2.0**: Can be downloaded [here](https://pbdl-ws.github.io/pbdl2021/challenge/download.html). To use the our DataLoader it is required to decompress the data and convert it to a hdf5-file (see [convert_hsds_to_hdf](https://github.com/nickstheisen/hyperseg/blob/main/hyperseg/datasets/pbdl_utils.py#L56)). 

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

|Dataset|Approach|Data|$R_\mu$|$R_M$|$F_{1_{M}}$|$J_M$|
|---|---|---|---|---|---|---|
|HCV2|U-Net|HSI                          |85.25|48.62|48.18|37.73|
| |RU-Net|HSI                            |87.63|54.14|53.26|43.33|
| |RU-Net|PCA1                           |88.25|58.07|55.43|44.26|
| |RU-Net|pRGB                           |87.95|56.65|55.46|44.03|
| |DL3+|HSI                              |86.60|53.15|51.83|40.79|
| |DL3+|PCA1                             |86.64|54.46|52.90|41.58|
| |DL3+|pRGB                             |87.00|55.33|54.08|42.58|
| |DL3+(BB)|pRGB                         |**90.26**|**64.10**|**61.93**|**50.04**|
| |(*DL3+(BB)*|*RGB*                      |*91.22*|*65.87*|*63.33*|*52.11*)|
| |DL3+(PT)|pRGB                         |89.62|61.91|60.17|48.47
|HyKo2|U-Net|HSI                         |85.36|68.15|68.55|57.39|
| |RU-Net|HSI                            |86.72|68.79|69.19|58.64|
| |RU-Net|PCA1                           |85.61|68.09|70.01|58.67|
| |RU-Net|pRGB                           |89.18|73.92|75.04|64.67|
| |DL3+|HSI                              |84.10|63.01|64.90|53.22|
| |DL3+|PCA1                             |79.99|61.59|63.00|50.40|
| |DL3+|pRGB                             |84.64|65.30|66.56|54.82|
| |DL3+(BB)|pRGB                         |**90.49**|**74.87**|**77.11**|**66.77**|
| |DL3+(PT)|pRGB                         |88.62|73.97|76.79|65.41|
|HSI-Drive|U-Net|HSI                     |94.95|74.74|76.08|64.95|
| |RU-Net|HSI                            |96.08|79.82|82.34|72.18|
| |RU-Net|PCA1                           |97.02|**86.80**|**87.76**|**79.23**|
| |RU-Net|pRGB                           |96.32|82.70|84.91|75.31|
| |DL3+|HSI                              |92.51|65.58|67.86|56.63|
| |DL3+|PCA1                             |90.88|62.93|64.31|52.62|
| |DL3+|pRGB                             |92.74|66.59|69.46|57.84|
| |DL3+(BB)|pRGB                         |**97.09**|83.93|86.41|77.44|
| |DL3+(PT)|pRGB                         |95.69|81.95|84.09|73.84|
| | |                                    |    |    |    |    |
|**Average </br> Performance**|U-Net|HSI |88.52|63.84|64.27|53.36|
| |RU-Net|HSI                            |90.14|67.58|68.26|57.68|
| |RU-Net|PCA1                           |90.29|70.99|71.07|60.72|
| |RU-Net|pRGB                           |91.15|71.09|71.80|61.34|
| |DL3+|HSI                              |87.74|60.58|61.53|50.21|
| |DL3+|PCA1                             |85.84|59.66|60.07|48.20|
| |DL3+|pRGB                             |88.13|62.41|63.37|51.75|
| |DL3+(BB)|pRGB                         |**92.61**|**74.30**|**75.15**|**64.75**|
| |DL3+(PT)|pRGB                         |91.31|72.61|73.68|62.57|
|**Worst-Case </br> Performance**|U-Net|HSI |82.25|48.63|48.18|37.73|
| |RU-Net|HSI                               |86.72|54.14|53.26|42.23| 
| |RU-Net|PCA1                              |85.61|58.07|55.43|44.26|
| |RU-Net|pRGB                              |87.95|56.65|55.46|44.03|
| |DL3+|HSI                                 |84.10|53.15|51.83|40.79|
| |DL3+|PCA1                                |79.99|54.46|52.90|41.58|
| |DL3+|pRGB                                |84.64|55.33|54.08|42.58|
| |DL3+(BB)|pRGB                            |**90.26**|**64.10**|**61.93**|**50.04**|
| |DL3+(PT)|pRGB                            |88.62|61.91|60.17|48.47|

The results highlighted in *italics* where achieved by using additional data from an RGB cam and are therefore surrounded in parenthesis. DL3+ stands for DeepLabV3+. BB suffix means that we used a MobileNetV2 backbone network pretrained on ImageNet and then finetuned on pseudo-RGB images derived from HSI. PT suffix means that we used a MobileNetV2 backbone network pretrained on CityScapes without further fine-tuning. all layers except for output-layer were frozen. 

## Reference
If you use our code please reference our paper *(Bibtex to final published version will be provided as soon as possible)*.
```
@misc{theisen2024hs3benchbenchmarkstrongbaseline,
      title={HS3-Bench: A Benchmark and Strong Baseline for Hyperspectral Semantic Segmentation in Driving Scenarios}, 
      author={Nick Theisen and Robin Bartsch and Dietrich Paulus and Peer Neubert},
      year={2024},
      doi={https://doi.org/10.48550/arXiv.2409.11205},
      eprint={2409.11205},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2409.11205}, 
}
```
