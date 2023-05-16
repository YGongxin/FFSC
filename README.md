# FFSC
FFSC：“Deep Sample Clustering Domain Adaption for Breast Histopathology”
This is the implementation of Deep Sample Clustering Domain Adaption for Breast Histopathology in Pytorch.

# Getting Started
Installation
The code was tested on Ubuntu 18.04.5 under the environment below:
PyTorch 1.7
Torchvision
Python 3.8
numpy 1.21.2
tensorflow 2.8.0

# Dataset
Download Dataset
Download BreaKHis Dataset at https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/.

Crop the dataset image to a pixel size of 224 * 224 and remove images that do not contain cellular tissue.
Place it in the directory ./dataset by training set and test set.

# Train
For example, if you run an experiment on adaptation from SNL to BreakHis_200x,
```
python main.py --source SNL --target BreakHis_200x
```
