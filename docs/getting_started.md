# Prerequisites

**Please ensure you have prepared the environment and the SemanticKITTI dataset.**

# Train and Test

## Stage-1: Class-Agnostic Query Proposal
Train QPN with 2 GPUs 
```
python train_stage1.py CONFIG = projects/configs/AEFFSSC/qpn.py
GPUS = 2
```

Eval QPN with 2 GPUs
```
python train_stage1.py 
```
## Stage-2: Class-Specific Voxel Segmentation
Train AEFF-SSC with temporal information with 2 GPUs 
```
python train_stage2.py CONFIG = projects/configs/AEFFSSC/AEFFSSC.py
GPUS = 2
```

Eval AEFF-SSC with temporal information with 2 GPUs
```
python train_stage2.py
```
