# Res_Swin: Effective combination of ResNet and Swin Transformer for denoising of CT images

This repository is a master thesis project in Leiden University
## Overview

- models folder: Include all the models of experiments, models of Red CNN, TransUNet refer from other repository

- warmup_scheduler folder: Implement warmup schedueler (reference)

- main.py: Implementation file with mutiple setting parameters

- train.py: Overall training pipline, including creating dataset, train and test.

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
Furthermore, there are some main essential libraries for this project:
- CUDA version: 11.6
- python: 3.7.11
- torch: 1.10.2
- torchvision: 0.11.3
- cudatoolkit: 11.3.1
- cudnn: 8.2.1
- numpy: 1.21.2
- segmentation-models-pytorch
