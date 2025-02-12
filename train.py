from __future__ import print_function
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
# from utils.classifiers.fc_net import *
from utils.data_utils import get_CIFAR10_data
from utils.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from utils.solver import Solver
from utils.layers import *
from utils.vis_utils import vis_accuracy_loss
import os

plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

data_dir = 'datasets'
if not os.listdir(data_dir):
    print('Please download the data from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz and put it under the datasets folder.')

# Load the (preprocessed) CIFAR10 data.
cifar10_dir = 'datasets/cifar-10-batches-py'

data = get_CIFAR10_data(cifar10_dir)
for k, v in list(data.items()):
    print(('%s: ' % k, v.shape))


class ConvNet(object):
    """
    A simple convolutional network with the following architecture:

    [conv - bn - relu] x M - global_average_pooling - affine - softmax
    
    For each convolution layer, downsampling of factor 2 is used by setting the stride
    to be 2. This gives a larger receptive field size.

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=[32], filter_sizes=[7],
            num_classes=10, weight_scale=1e-3, reg=0.0, use_batch_norm=True, 
            dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer. It is a
          list whose length defines the number of convolution layers
        - filter_sizes: Width/height of filters to use in the convolutional layer. It
          is a list with the same length with num_filters
        - num_classes: Number of output classes
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - use_batch_norm: A boolean variable indicating whether to use batch normalization
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        
        assert len(num_filters) == len(filter_sizes)
        
        self.M = len(num_filters)
        C, H, W = input_dim
        self.use_batch_norm = use_batch_norm
        self.bn_params = {}

        for i in range(self.M):
            F = num_filters[i]
            if i > 0: C = num_filters[i-1]
            HH = filter_sizes[i]
            WW = HH
            self.params[f'W{i+1}'] = np.random.randn(F, C, HH, WW) * weight_scale
            self.params[f'b{i+1}'] = np.zeros(F, dtype=self.dtype)

            if self.use_batch_norm:
                self.params[f'gamma{i+1}'] = np.ones(F, dtype=self.dtype)
                self.params[f'beta{i+1}'] = np.zeros(F, dtype=self.dtype)
                self.bn_params[f'conv{i+1}'] = {'mode': 'train', 'eps': 1e-5, 'momentum': 0.9}

        self.params['W_affine'] = np.random.randn(num_filters[-1], num_classes) * weight_scale
        self.params['b_affine'] = np.zeros(num_classes, dtype=self.dtype)

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the simple convolutional network.

        Inputs:
        - X: Array of input data of shape (N, C, H, W)bn_paramX[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """

        scores = None
        mode = 'test' if y is None else 'train'
        
        # Forward pass
        outs = {}
        caches = {}
        
        for i in range(self.M):
            self.bn_params[f'conv{i+1}']['mode'] = mode

            if(i == 0):
                input = X
            else:
                input = outs[f'relu{i}']

            outs[f'conv{i+1}'], caches[f'conv{i+1}'] = conv_forward_naive(input, self.params[f'W{i+1}'], self.params[f'b{i+1}'],
                                                                          {'stride': 2, 'pad': (self.params[f'W{i+1}'].shape[2]-1)//2})
            outs[f'bn{i+1}'], caches[f'bn{i+1}'] = spatial_batchnorm_forward(outs[f'conv{i+1}'], self.params[f'gamma{i+1}'], 
                                                                             self.params[f'beta{i+1}'], self.bn_params[f"conv{i+1}"])
            outs[f'relu{i+1}'], caches[f'relu{i+1}'] = relu_forward(outs[f'bn{i+1}'])

        outs['global_avg_pool'], caches['global_avg_pool'] = global_avg_pool_forward(outs[f'relu{self.M}'])

        outs['affine'], caches['affine'] = affine_forward(outs['global_avg_pool'], self.params['W_affine'], self.params['b_affine'])

        scores = outs['affine']

        if y is None:
            return scores

        # Backward pass
        loss, grads = 0, {}
        
        loss, dout = softmax_loss(scores, y)

        loss += 0.5 * self.reg * np.sum(self.params['W_affine']**2)
        dx, grads['W_affine'], grads['b_affine'] = affine_backward(dout, caches['affine'])
        grads['W_affine'] += self.reg * self.params['W_affine']

        dx = global_avg_pool_backward(dx, caches['global_avg_pool'])
        # print(f"dx shape: {dx.shape}")

        for i in range(self.M, 0, -1):
            loss += 0.5 * self.reg * np.sum(self.params[f'W{i}']**2)

            # print(f"dx shape: {dx.shape}")
            # print(f"relu shape: {outs[f'relu{i}'].shape}")

            dx = relu_backward(dx, caches[f'relu{i}'])

            if self.use_batch_norm:
                dx, grads[f'gamma{i}'], grads[f'beta{i}'] = spatial_batchnorm_backward(dx, caches[f'bn{i}'])
            
            dx, grads[f'W{i}'], grads[f'b{i}'] = conv_backward_naive(dx, caches[f'conv{i}'])

            grads[f'W{i}'] += self.reg * self.params[f'W{i}']

        return loss, grads
    
def overfit_check():
    '''
    Here, due to the limited training epochs, we are not really able to overfit 
    the training data with a 100% training accuracy. I get around 50% training 
    accuracy and 19% validation accuracy.
    '''
    np.random.seed(231)
    num_train = 100
    small_data = {
    'X_train': data['X_train'][:num_train],
    'y_train': data['y_train'][:num_train],
    'X_val': data['X_val'],
    'y_val': data['y_val'],
    }

    model = ConvNet(
        num_filters=[16, 32],
        filter_sizes=[7, 3],
        weight_scale=1e-2
    )

    solver = Solver(
        model, small_data,
        num_epochs=100, batch_size=20,
        update_rule='sgd_momentum',
        optim_config={
        'learning_rate': 1e-2,
        },
        verbose=True, print_every=10
    )
    solver.train()
    vis_accuracy_loss(solver)

def main():
    ''' Configure and train the network '''

    model = ConvNet(
        num_filters=[16, 32, 64, 128],
        filter_sizes=[7, 3, 3, 3],
        weight_scale=1e-2,
        reg=0.00,
        use_batch_norm=True
    )

    solver = Solver(
        model, data,
        num_epochs=3, batch_size=50,
        update_rule='sgd_momentum',
        optim_config={
        'learning_rate': 5e-2,
        },
        lr_decay=0.5,
        verbose=True, print_every=50
    )

    solver.train()

    trained_model = model
    trained_model.params.update(solver.best_params)

    # Run the model on validation and test sets
    y_test_pred = np.argmax(trained_model.loss(data['X_test']), axis=1)
    y_val_pred = np.argmax(trained_model.loss(data['X_val']), axis=1)
    print('Validation set accuracy: ', (y_val_pred == data['y_val']).mean())
    print('Test set accuracy: ', (y_test_pred == data['y_test']).mean())

    solver.checkpoint_name = 'saved_models/latest_model'
    solver._save_checkpoint()

if __name__ == '__main__':
    main()
