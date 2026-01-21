<div align="center">

# [SPVIS: Enhancing Video Instance Segmentation through Stabilized Feature Propagation](https://arxiv.org/abs/2306.03413)
Yan Jin, Fang Gao, Qingjiao Meng, Qingbao Huang,Hanbo Zheng, Shengheng Ma



<img src="https://github.com/zhang-tao-whu/paper_images/blob/master/dvis/pipeline.png" width="800"/>
</div>

## Abstract
There has been remarkable progress on object detection and re-identification in recent years which are the core components for multi-object tracking. However, little attention has been focused on accomplishing the two tasks in a single network to improve the inference speed. The initial attempts along this path ended up with degraded results mainly because the re-identification branch is not appropriately learned. In this work, we study the essential reasons behind the failure, and accordingly present a simple baseline to addresses the problems. It remarkably outperforms the state-of-the-arts on the MOT challenge datasets at 30 FPS. We hope this baseline could inspire and help evaluate new ideas in this field.


## Features
-	SPVIS Framework: Novel feature propagation method for video instance segmentation that mitigates temporal error accumulation and feature degradation.
-	Dual-component Architecture: Proposes an integrated architecture of Progressive Tracker (PGT) and Joint Feature-Preserving Modeling for stable long-term propagation with computational efficiency.. 
-	SOTA Performance: Achieves 69.5/64.6/51.9/54.3 AP on YouTubeVIS 2019/2021/2022 and OVIS benchmarks, balancing accuracy and efficiency.
-	Versatile Deployment: Effective in both online/offline settings for challenging scenarios (long sequences, low frame rates, heavy occlusion). 

## Tracking performance
### Results on MOT challenge test set
| Dataset    |   AP   | & AP$_{50}$ | AP$_{75}$ | AR$_{1}$ |AR$_{10}$ | FPS |
|--------------|-----------|--------|-------|----------|----------|--------|
|Youtube-VIS 2019  | 60.6 | 64.7 |  591 | 47.6% | 11.0% | 30.5 |
|Youtube-VIS 2021       | 74.9 | 72.8 | 1074 | 44.7% | 15.9% | 25.9 |
|Youtube-VIS 2022       | 73.7 | 72.3 | 3303 | 43.2% | 17.3% | 25.9 |
|OVIS      | 61.8 | 67.3 | 5243 | 68.8% | 7.6% | 13.2 |

 All of the results are obtained on the [MOT challenge](https://motchallenge.net) evaluation server under the “private detector” protocol. We rank first among all the trackers on 2DMOT15, MOT16, MOT17 and  MOT20. The tracking speed of the entire system can reach up to **30 FPS**.

## Installation

### Requirements
- Python ≥ 3.6
- PyTorch ≥ 1.9 and [torchvision](https://github.com/pytorch/vision/) that matches the PyTorch installation.
  Install them together at [pytorch.org](https://pytorch.org) to make sure of this. Note, please check
  PyTorch version matches that is required by Detectron2.
- Detectron2: follow [Detectron2 installation instructions](https://detectron2.readthedocs.io/tutorials/install.html).
- OpenCV is optional but needed by demo and visualization
- Panopticapi: `pip install git+https://github.com/cocodataset/panopticapi.git`
- `pip install -r requirements.txt`

### CUDA kernel for MSDeformAttn
After preparing the required environment, run the following command to compile CUDA kernel for MSDeformAttn:

`CUDA_HOME` must be defined and points to the directory of the installed CUDA toolkit.

```bash
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
```

### Example conda environment setup
```bash
conda create --name SPVIS python=3.8 -y
conda activate SPVIS
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -U opencv-python

# install detectron2
python -m pip install detectron2 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html

# install panoptic api
pip install git+https://github.com/cocodataset/panopticapi.git


# Prepare Datasets for DVIS

A dataset can be used by accessing [DatasetCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.DatasetCatalog)
for its data, or [MetadataCatalog](https://detectron2.readthedocs.io/modules/data.html#detectron2.data.MetadataCatalog) for its metadata (class names, etc).
This document explains how to setup the builtin datasets so they can be used by the above APIs.
[Use Custom Datasets](https://detectron2.readthedocs.io/tutorials/datasets.html) gives a deeper dive on how to use `DatasetCatalog` and `MetadataCatalog`,
and how to add new datasets to them.

DVIS has builtin support for a few datasets.
The datasets are assumed to exist in a directory specified by the environment variable
`DETECTRON2_DATASETS`.
Under this directory, detectron2 will look for datasets in the structure described below, if needed.
```
$DETECTRON2_DATASETS/
  ytvis_2019/
  ytvis_2021/
  ovis/
  VIPSeg/
  VSPW_480p/
```

You can set the location for builtin datasets by `export DETECTRON2_DATASETS=/path/to/datasets`.
If left unset, the default is `./datasets` relative to your current working directory.

The [model zoo](../MODEL_ZOO.md)
contains configs and models that use these builtin datasets.


## Expected dataset structure for [YouTubeVIS 2019](https://competitions.codalab.org/competitions/20128):

```
ytvis_2019/
  {train,valid,test}.json
  {train,valid,test}/
    Annotations/
    JPEGImages/
```

## Expected dataset structure for [YouTubeVIS 2021](https://competitions.codalab.org/competitions/28988):

```
ytvis_2021/
  {train,valid,test}.json
  {train,valid,test}/
    Annotations/
    JPEGImages/
```

## Expected dataset structure for [Occluded VIS](http://songbai.site/ovis/):

```
ovis/
  annotations/
    annotations_{train,valid,test}.json
  {train,valid,test}/
```
## Expected dataset structure for [VIPSeg](https://github.com/VIPSeg-Dataset/VIPSeg-Dataset):

After downloading the VIPSeg dataset, it still needs to be processed according to the official script. To save time, you can directly download the processed VIPSeg dataset from [baiduyun](https://pan.baidu.com/s/1SMausnr6pVDJXTGISeFMuw) (password is `dvis`). 
```
VIPSeg/
  VIPSeg_720P/
    images/
    panomasksRGB/
    panoptic_gt_VIPSeg_{train,val,test}.json
```

## Expected dataset structure for [VSPW](https://codalab.lisn.upsaclay.fr/competitions/7869#participate):

```
VSPW_480p/
  data/
  {train,val,test}.txt
```

## Register your own dataset:

- If it is a VIS/VPS/VSS dataset, convert it to YTVIS/VIPSeg/VSPW format. If it is a image instance dataset, convert it to COCO format.
- Register it in `/dvis/data_video/datasets/{builtin,vps,vss}.py`

```BibTeX
@article{DVIS,
  title={DVIS: Decoupled Video Instance Segmentation Framework},
  author={Zhang, Tao and Tian, Xingye and Wu, Yu and Ji, Shunping and Wang, Xuebo and Zhang, Yuan and Wan, Pengfei},
  journal={arXiv preprint arXiv:2306.03413},
  year={2023}
}

@article{zhang2023vis1st,
  title={1st Place Solution for the 5th LSVOS Challenge: Video Instance Segmentation},
  author={Zhang, Tao and Tian, Xingye and Zhou, Yikang and Wu, Yu and Ji, Shunping and Yan, Cilin and Wang, Xuebo and Tao, Xin and Zhang, Yuan and Wan, Pengfei},
  journal={arXiv preprint arXiv:2308.14392},
  year={2023}
}

@article{zhang2023vps1st,
  title={1st Place Solution for PVUW Challenge 2023: Video Panoptic Segmentation},
  author={Zhang, Tao and Tian, Xingye and Wei, Haoran and Wu, Yu and Ji, Shunping and Wang, Xuebo and Zhang, Yuan and Wan, Pengfei},
  journal={arXiv preprint arXiv:2306.04091},
  year={2023}
}
```

## Acknowledgement

This repo is largely based on [Mask2Former](https://github.com/facebookresearch/Mask2Former), [MinVIS](https://github.com/NVlabs/MinVIS) and [VITA](https://github.com/sukjunhwang/VITA).
Thanks for their excellent works.
