# NCP (Neighborhood Conformal Prediction)
This repository contains the code and models necessary to replicate the results of our recent paper:-
**Improving Uncertainty Quantification of Deep Classifiers via Neighborhood Conformal Prediction: Novel Algorithm and Theoretical Analysis** <br>
Please follow the project page [here](https://sites.google.com/view/ncpefficiency/home) for new updates.
## Contents
The major content of our repo are:
 - `CP/` The main folder containing the python scripts for running the experiments.
 - `checkpoints/` Our pre trained models.
 - `datasets/` A folder that contains the datasets used in our experiments CIFAR10, CIFAR100, Imagenet.
 - `Results/` A folder that contains different files from different experiments.

CP folder contains:

1. `Classification.py`: the main code for running experiments.

## Prerequisites

Prerequisites for running our code:
 - numpy
 - scipy
 - sklearn
 - torch
 - tqdm
 - seaborn
 - torchvision
 - pandas
 - plotnine
 
## Running instructions
1.  Install dependencies:
```
conda create -n NCP python=3.8
conda activate NCP
conda install -c conda-forge numpy
conda install -c conda-forge scipy
conda install -c conda-forge scikit-learn
conda install -c conda-forge tqdm
conda install -c conda-forge seaborn
conda install -c conda-forge pandas
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
conda install -c conda-forge plotnine
```
2. 
i. Download our trained models from [here](https://drive.google.com/drive/folders/1KwkzR7cXc3QWexKjcbUG1-ZWeMqzg5Nl?usp=share_link) and extract them to NCP/checkpoints.

ii. If you want to run ImageNet experiments, obtain a copy of ImageNet ILSVRC2012 validation set from [here](https://berkeley.box.com/shared/static/pouthcomrvxw9hj64oxhacjvqdw3ihlp.gz) and extract the data. Put the created folders in /datasets/.

3. The current working directory when running the scripts should be the folder NCP.


To reproduce the results for CIFAR10:
```
python -W ignore ClassificationTestMyFunctions.py --dataset CIFAR10
python -W ignore plot_classification.py --dataset CIFAR10

```


To reproduce the results for CIFAR100:
```
python -W ignore ClassificationTestMyFunctions.py --dataset CIFAR100
python -W ignore plot_classification.py --dataset CIFAR100
```

To reproduce the results for ImageNet:
```
python -W ignore ClassificationTestMyFunctions.py --dataset ImageNet
python -W ignore plot_classification.py --dataset ImageNet

```
