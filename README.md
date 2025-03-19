# L361-FL-project

This project aims to define the concept of critical batch size in a federated learning setting. This work is inspired by (https://arxiv.org/abs/1812.06162), where in centralized machine learning paradigm critical batch size is defined to locate the point where diminishing returns in terms of training speed start to emerge when increasing batch size. We define an analogous concept in federated learning, showing that critical batch size in federated learning shows a point where diminishing returns appear when increasing global batch size.

## Quick start:

### 1. Create a conda environment with python 3.11.11 and install the dependencies:
```
conda create -n flbs python=3.11.11
conda activate flbs

pip install -r requirements.txt
```

## Repo structure

Notebooks 
`main_noniid.ipynb` shows the experiments that generated scaling laws for federated learning with non-IID data, as well as estimates the critical batch size.
`main_iid.ipynb` serves the same purpose as the `main_noniid.ipynb` notebook, showing the same results for IID data. 

`src` folder contains necessary boilerplate code needed to run experiments with [Flower](https://flower.ai/) framework, as well as some scripts for plotting the scaling laws in a prettier manner.

`notebooks` contains other necessary notebooks (for example centralized baseline scaling laws and critical batch size estimation).
