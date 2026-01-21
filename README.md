
# [SPVIS: Enhancing Video Instance Segmentation through Stabilized Feature Propagation]
Yan Jin, Fang Gao, Qingjiao Meng, Qingbao Huang,Hanbo Zheng, Shengheng Ma


## :sunny: Structure of SPVIS
<img src="utils/1.png">



## Abstract
Video instance segmentation (VIS) extends instance-level understanding from static images to continuous video, necessitating accurate pixel-level masks and consistent identity association across frames. Feature propagation approaches, while computationally efficient, are often hindered by error accumulation and feature degradation over time. We introduce SPVIS, a VIS framework based on feature propagation, addressing these challenges through in-memory object-query propagation. SPVIS comprises a Progressive Tracker (PGT) for cross-clip association with error correction and joint feature-preserving modeling, including the Refinement Compensator (RCP) and Spatial Interaction Module (SIM), to maintain high-quality object queries. Across standard benchmarks, SPVIS achieves competitive accuracy-efficiency trade-offs, delivering 69.5, 64.6, 51.9, and 54.3 AP on YouTube-VIS 2019, 2021, 2022, and OVIS, respectively. Our framework provides a lightweight solution for long-sequence association, including scenarios with low frame rates and occlusions.


## Features
-	SPVIS Framework: Novel feature propagation method for video instance segmentation that mitigates temporal error accumulation and feature degradation.
-	Dual-component Architecture: Proposes an integrated architecture of Progressive Tracker (PGT) and Joint Feature-Preserving Modeling for stable long-term propagation with computational efficiency.. 
-	SOTA Performance: Achieves 69.5/64.6/51.9/54.3 AP on YouTubeVIS 2019/2021/2022 and OVIS benchmarks, balancing accuracy and efficiency.
-	Versatile Deployment: Effective in both online/offline settings for challenging scenarios (long sequences, low frame rates, heavy occlusion). 

## Tracking performance
### Results on Youtube-VIS challenge test set
| Dataset    |   AP   | &AP_{50}$ | $AP_{75}$ | $AR_{1}$ |$AR_{10}$ |
|--------------|-----------|--------|-------|----------|----------|
|Youtube-VIS 2019  | 69.5 | 92.0 |  77.8 | 61.7 | 75.9 |
|Youtube-VIS 2021       | 64.6 | 86.9 | 72.1 | 49.6 | 70.5 |
|Youtube-VIS 2022       | 51.9 | 73.2 | 54.7 | 41.5 | 56.6 |
|OVIS      | 54.3 | 78.9 | 59.3 | 20.9 | 59.9 |


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

```

### Dataset Preparation
- Refer to the dataset preparation method of [DVIS](https://github.com/zhang-tao-whu/DVIS/blob/main/datasets/README.md).

```BibTeX

@article{jinVIS2023-3st,
  title={SPVIS: Enhancing Video Instance Segmentation through Stabilized Feature Propagation},
  author={Yan Jin, Fang Gao, Qingjiao Meng, Qingbao Huang,Hanbo Zheng, Shengheng Ma},
  journal={The Visual Computer},
  year={2026}
}
```

## Acknowledgement

This repo is largely based on [Mask2Former](https://github.com/facebookresearch/Mask2Former), [MinVIS](https://github.com/NVlabs/MinVIS) and [VITA](https://github.com/sukjunhwang/VITA), [DVIS](https://github.com/zhang-tao-whu/DVIS).
Thanks for their excellent works.
