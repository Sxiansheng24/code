<div align="center">   
  
# AEFF-SSC: An Attention-Enhanced Feature Fusion for 3D Semantic Scene Completion
</div>




## Getting Started
- [Installation](docs/install.md) 
- [Prepare Dataset](docs/prepare_dataset.md)
- [Run and Eval](docs/getting_started.md)

## Model Zoo
The query proposal network (QPN) for stage-1 is available [here](https://pan.baidu.com/s/1oNsGxPyWcfLI_4cESsfDHA?pwd=303i).
For stage-2, please download the trained models based on the following table.

| Backbone |   Method   | Lr Schd |  IoU  | mIoU  | Config | Download |
| :---: |:----------:|:-------:|:-----:|:-----:| :---: | :---: |
| [R50](https://drive.google.com) | AEFF-SSC+  |  24ep   | 45.32 | 14.00 |[config](projects/configs/AEFFSSC/AEFFSSC+.py) |[model](https://pan.baidu.com/s/1oNsGxPyWcfLI_4cESsfDHA?pwd=303i) |
| [R50](https://drive.google.com) |  AEFF-SSC  |  24ep   | 44.72 | 13.20 |[config](projects/configs/AEFFSSC/AEFFSSC.py) |[model](https://pan.baidu.com/s/1oNsGxPyWcfLI_4cESsfDHA?pwd=303i)|


 
## Dataset

- [x] SemanticKITTI
