from builtins import range
from builtins import object
import os
import numpy as np
from torch import softmax

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dim)
        self.params['b1'] = np.zeros(hidden_dim)
        self.params['W2'] = weight_scale * np.random.randn(hidden_dim, num_classes)
        self.params['b2'] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

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
        # layer1 affine & relu
        out, relu_cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        # layer2 affine
        scores, fc = affine_forward(out, self.params['W2'], self.params['b2'])

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        # layer2 softmax
        # Numerical stability: subtract max from each row
        scores =scores - np.max(out, axis=1, keepdims=True) # (N, C)

        # Compute softmax probabilities: (N, C)
        scores = np.exp(scores)
        probs = scores / np.sum(scores, axis=1, keepdims=True)
        log_probs = -np.log(probs)
        loss, grads = 0, {}
        num_train = X.shape[0]
        correct_log_probs = log_probs[np.arange(num_train), y] # shape (N)
        loss = np.mean(correct_log_probs) +\
          0.5 * self.reg * np.sum(self.params['W1'] ** 2) +\
          0.5 * self.reg * np.sum(self.params['W2'] ** 2)

        # cross-androphy backward
        dout = probs.copy()  # (N, C)
        dout[np.arange(num_train), y] -= 1  # subtract 1 at correct class positions
        dout = dout / num_train

        dout, grads['W2'], grads['b2'] = affine_backward(dout, fc)
        grads['W2'] += self.reg * self.params['W2']
        dout, grads['W1'], grads['b1'] = affine_relu_backward(dout, relu_cache)
        grads['W1'] += self.reg * self.params['W1']
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True



class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        
        current_input_dim = input_dim
        for idx in range(1,self.num_layers):
            hidden_dim = hidden_dims[idx-1]
            self.params['W'+str(idx)] = weight_scale * np.random.randn(current_input_dim, hidden_dim)
            self.params['b'+str(idx)] = np.zeros(hidden_dim)
            current_input_dim = hidden_dim
            if normalization and idx < self.num_layers-1:
              self.params['gamma'+str(idx)] = np.ones(hidden_dim)
              self.params['beta'+str(idx)] = np.zeros(hidden_dim)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
          # deault mode of batch normalization is 'train', 
          # TODO whether the last layer will be used or not
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

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
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        input_nums = X.shape[0]
        current_input = X
        caches = []
        for idx in range(1,self.num_layers):
            cache = {}
            W = self.params['W'+str(idx)]
            b = self.params['b'+str(idx)]
            current_input, af_cache = affine_forward(current_input, W, b)
            cache['af_cache'] = af_cache
            if idx != self.num_layers:

              # l2 normalization
              if self.normalization == 'batchnorm':
                gamma = self.params['gamma'+str(idx)]
                beta = self.params['beta'+str(idx)]
                current_input, bf_cache = batchnorm_forward(current_input, gamma, beta, self.bn_params[idx-1])
                cache['bf_cache'] = bf_cache

              # Relu
              current_input, rf_cache = relu_forward(current_input)
              cache['rf_cache'] = rf_cache

              # dropout
              if self.use_dropout:
                current_input, df_cache = dropout_forward(current_input, self.dropout_param)
                cache['df_cache'] = df_cache

            # else do nothing
            caches.append(cache)
        scores = current_input

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################

        loss, dout = softmax_loss(scores, y)
        # l2 regularization
        for idx in range(1,self.num_layers):
            W = self.params['W'+str(idx)]
            loss += 0.5 * self.reg * np.sum(W**2)
          
        for idx in range(self.num_layers-1, 0,-1):

            cache = caches[idx-1]
            if idx != self.num_layers:

              # dropout backward
              if self.use_dropout:
                df_cache = cache['df_cache']
                dout = dropout_backward(dout, df_cache)
                
              # Relf backward
              rf_cache = cache['rf_cache']
              dout = relu_backward(dout, rf_cache)

              # L2 backward
              if self.normalization == 'batchnorm':
                bf_cache = cache['bf_cache']
                dout = batchnorm_backward(dout, bf_cache)
              
              af_cache = cache['af_cache']
              dout, dw, db = affine_backward(dout, af_cache)
              grads['W'+str(idx)] = dw + self.reg * self.params['W'+str(idx)]
              grads['b'+str(idx)] = db
            
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


    def save(self, fname):
      """Save model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      params = self.params
      np.save(fpath, params)
      print(fname, "saved.")
    
    def load(self, fname):
      """Load model parameters."""
      fpath = os.path.join(os.path.dirname(__file__), "../saved/", fname)
      if not os.path.exists(fpath):
        print(fname, "not available.")
        return False
      else:
        params = np.load(fpath, allow_pickle=True).item()
        self.params = params
        print(fname, "loaded.")
        return True