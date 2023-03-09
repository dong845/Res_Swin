# Res-Swin: Effective Combination of ResNet and Swin Transformer for LDCT Denoising

This is my master thesis project in the Leiden University

## Overview

- models folder: Include all the models of experiments, among them, RED-CNN and TransUNet refer from other repositories

- warmup_scheduler folder: Implement schedueler with warmup period (reference)

- main.py: File with mutiple setting parameters, and can be used for training with different dataset and visualize saved results

- train.py: Overall training pipline, including creating dataset, training process and testing process.

- measure.py: Measurement function of PSNR and SSIM (reference)

- visualize.py: Visualization of images gotten from models to see more details and measure the performance.

- Running with default settings:
```
python main.py
```

## Requirements
The file requirements imports all the required libraries, but there are also some libraries that are not related to this project. You can install them by:
```
pip install -r requirements.txt
```
Note: There are some main essential settings for this project:
- CUDA version: 11.6
- python: 3.7.11
- torch: 1.10.2
- torchvision: 0.11.3
- cudatoolkit: 11.3.1
- cudnn: 8.2.1
- numpy: 1.21.2
- albumentations: 1.1.0
- segmentation-models-pytorch
