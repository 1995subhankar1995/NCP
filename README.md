# NCP (Neighborhood Conformal Prediction)
This repository contains the code and models necessary to replicate the results of our recent paper:
**Neighborhood Conformal Prediction** <br>

## Contents
The major content of our repo are:
 - `RSCP/` The main folder containing the python scripts for running the experiments.
 - `third_party/` Third-party python scripts imported. Specifically we make use of the SMOOTHADV attack by [Salman et al (2019)](https://github.com/Hadisalman/smoothing-adversarial)
 - `Create_Figures/` Python scripts for creating all the figures in the paper. The /Create_Figures/Figures subfolder contains the figures themselves.
 - `Arcitectures/` Architectures for our trained models.
 - `Pretrained models/` Cohen pretrained models. [Cohen et al (2019)](https://github.com/locuslab/smoothing)
 - `checkpoints/` Our pre trained models.
 - `datasets/` A folder that contains the datasets used in our experiments CIFAR10, CIFAR100, Imagenet.
 - `Results/` A folder that contains different csv files from different experiments, used to generate the results in the paper.

RSCP folder contains:

1. `RSCP_exp.py`: the main code for running experiments.
2. `Score_Functions.py`: containing all non-conformity scores used.
3. `utills.py`: calibration and predictions functions, as well as other function used in the main code.

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
   1. Download our trained models from [here](https://drive.google.com/drive/folders/1KwkzR7cXc3QWexKjcbUG1-ZWeMqzg5Nl?usp=share_link) and extract them to Project_RSCP/checkpoints/.
   2. Download cohen models from [here](https://drive.google.com/file/d/1h_TpbXm5haY5f-l4--IKylmdz6tvPoR4/view) and extract them to Project_RSCP/Pretrained_Models/. Change the name of "models" folder to "Cohen".
   3. If you want to run ImageNet experiments, obtain a copy of ImageNet ILSVRC2012 validation set and preprocess the val directory by running [this script](https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh). Put the created folders in Project_RSCP/datasets/imagenet/. 
   4. Optional: download our pre created adversarial examples from [here](https://technionmail-my.sharepoint.com/:f:/g/personal/asafgendler_campus_technion_ac_il/Es1JTaMEdMZEhG480b_qjcYBo6znBVS5FKrOewMjVw0NNw?e=hcbkag) and extract them to Project_RSCP/Adversarial_Examples/.

3. The current working directory when running the scripts should be the top folder RSCP.


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
