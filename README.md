# pytorch-fcn
This is a USC CSCI677 programming assignment to create, train, and test a CNN for the task of semantic segmentation FCN32, FCN16 method.

## Usage in Notebook / Colab
```
from fcn import train
train("FCN32", epoch=100) # train FCN32 for 100 epoches
train("FCN16", epoch=100) # train FCN16 for 100 epoches
```
The hyperparameters are fixed due to homework requirement, but feel free to go through the code and do some changes.
