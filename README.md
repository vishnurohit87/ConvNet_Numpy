A simple Convolutional Neural Network (CNN) written from scratch using Numpy. Trained on CIFAR10 dataset. 


Architecture: [conv - bn - relu] x M - global_average_pooling - affine - softmax.

Worked on this project to understand the working of each individual compoent in a CNN. The implementation all the layers used in the network can be found in utils/layers.py

Current configurations yield around 65% test set accuracy with 4 convolution layers (M=4).

### Setup

Create and activate the conda environment using appropriate command. On Windows, open the installed "Conda prompt" to run the command. On MacOS and Linux, you can just use a terminal window to run the command, Run: `conda env create -f env_cnn_numpy.yml`
