import numpy as np

def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = x.reshape((x.shape[0], -1)).dot(w) + b
    cache = (x, w, b)
    return out, cache

def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx = dout.dot(w.T).reshape(cache[0].shape)
    dw = cache[0].reshape((cache[0].shape[0], -1)).T.dot(dout)
    db = np.sum(dout, axis=0)
    return dx, dw, db

def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = np.maximum(x, 0)
    cache = x
    return out, cache

def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    dx = dout * (x > 0)
    return dx

def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx

def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    H_out, W_out = compute_output_dims(x.shape, w.shape, conv_param)

    pad = conv_param['pad']
    stride = conv_param['stride']

    # Pad the input
    x_padded = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant', constant_values=0)
    
    
    # Debug Statements
    # print(f"Input shape: {x.shape}")
    # print(f"Filter shape: {w.shape}")
    # print(f"Output shape: {H_out, W_out}")
    # print(f"Input shape after padding: {x_padded.shape}")
    # print(f"Padding: {pad}")
    # print(f"W shape: {w.shape}")
    # print(f"Output shape: {H_out, W_out}")

    out = np.zeros((N, F, H_out, W_out))

    for i in range(W_out):
        for j in range(H_out):
            # Get the patch
            x_data = x_padded[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]
            
            # Reshape the patch
            x_data = x_data.reshape(N, -1)
            # print(f"Patch shape after reshaping: {x_data.shape}")

            out[:, :, i, j] = x_data.dot(w.reshape(F, -1).T) + b


    cache = (x, w, b, conv_param)
    return out, cache

def compute_output_dims(x_shape, w_shape, conv_param):
    """
    Compute the spatial dimensions of the output volume.

    Input:
    - x_shape: Tuple of (N, C, H, W)
    - w_shape: Tuple of (F, C, HH, WW)
    - conv_param: Dictionary of convolution layer parameters

    Returns a tuple of:
    - H_out: Output height
    - W_out: Output width
    """

    H_out = 1 + (x_shape[2] + 2 * conv_param['pad'] - w_shape[2]) // conv_param['stride']
    W_out = 1 + (x_shape[3] + 2 * conv_param['pad'] - w_shape[3]) // conv_param['stride']
    
    return np.ceil(H_out).astype(int), np.ceil(W_out).astype(int)

def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    
    x, w, b, conv_param = cache
    N, C, H, W = x.shape
    F, C, HH, WW = w.shape
    N, F, H_out, W_out = dout.shape

    pad = conv_param['pad']
    stride = conv_param['stride']

    x_padded = np.pad(x, ((0,0), (0,0), (pad,pad), (pad,pad)), 'constant', constant_values=0)
    dx = np.zeros_like(x)
    dx_padded = np.zeros_like(x_padded) 
    dw, db = np.zeros_like(w), np.zeros_like(b)
    # print(f"dout shape: {dout.shape}")

    for i in range(W_out):
        for j in range(H_out):
            # Get the patch
            x_data = x_padded[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW]

            dout_reshape = dout[:, : , i, j].T.reshape(F, -1)
            dx_padded[:, :, i*stride:i*stride+HH, j*stride:j*stride+WW] += dout_reshape.T.dot(w.reshape(F, -1)).reshape(N, C, HH, WW)

            dw += x_data.T.dot(dout[:, :, i, j]).T.reshape(F, C, HH, WW)

            db += np.sum(dout[:, :, i, j], axis=0)
    
    dx = dx_padded[:, :, pad:-pad, pad:-pad]
    return dx, dw, db

def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None

    if mode == 'train':
        x_mean = np.mean(x, axis=0, keepdims=True)
        x_var = np.var(x, axis=0, keepdims=True)

        x_norm = (x - x_mean) / np.sqrt(x_var + eps)
        out = gamma * x_norm + beta

        running_mean = momentum * running_mean + (1 - momentum) * x_mean
        running_var = momentum * running_var + (1 - momentum) * x_var

        cache = (x, x_mean, x_var, x_norm, gamma, beta, eps)


    elif mode == 'test': 
        x_norm = (x- running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_norm + beta

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache

def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    N, C, H, W = x.shape
    xT = x.transpose(0,2,3,1).reshape(-1, C)
    outT, cache = batchnorm_forward(xT, gamma, beta, bn_param)
    out = outT.reshape(N, H, W, C).transpose(0,3,1,2)
    
    return out, cache

def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None

    x, x_mean, x_var, x_norm, gamma, beta, eps = cache
    N, D = x.shape

    dgamma = np.sum(dout * x_norm, axis=0)
    dbeta = np.sum(dout, axis=0)

    dx_norm = dout*gamma
    dvar = np.sum(dx_norm*(x-x_mean)*(-0.5)*((x_var+eps)**(-3/2)), axis=0)
    dmean = np.sum(dx_norm*(-1/(np.sqrt(x_var+eps))),axis=0) + dvar*(np.sum(-2*(x-x_mean),axis=0)/N)
    dx = dx_norm*(1/(np.sqrt(x_var+eps))) + dvar*2*(x-x_mean)/N + dmean/N

    return dx, dgamma, dbeta

def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    N, C, H, W = dout.shape
    doutT = dout.transpose(0,2,3,1).reshape(-1, C)
    dxT, dgamma, dbeta = batchnorm_backward(doutT, cache)
    dx = dxT.reshape(N, H, W, C).transpose(0,3,1,2)

    return dx, dgamma, dbeta

def global_avg_pool_forward(x):
    """
    Computes the forward pass of the global average pooling layer
    
    Input:
    - x: Input data of shape (N, C, H, W)
    
    Returns of a tuple of:
    - out: Output data, of shape (N, C)
    - cache: (x,)
    """
    out, cache = None, None

    out = np.mean(x, axis=(2,3))
    cache = (x,)

    return out, cache

def global_avg_pool_backward(dout, cache):
    """
    Computes the forward pass of the global average pooling layer
    
    Inputs:
    - dout: Upstream derivatives.
    - cache: x as in global_avg_pool_backward
    
    Returns:
    - dx: gradient with respect x
    
    """
    dx = None
    
    x, = cache
    N, C, H, W = x.shape
    dx = dout.reshape(N,C,1,1)
    dx = np.repeat(np.repeat(dx, H, axis=2), W, axis=3) / (H*W)

    return dx   
