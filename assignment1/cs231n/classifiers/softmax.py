from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    num_dims = W.shape[0]

    def _numeric_gradient(f):
      """
      no batch execute the gradient of function f at X.
      Input: 
      X: A numpy array with shape (D,)
      f: A function
      """
      h = 1e-5
      dw = np.zeros_like(W)
      for i in range(W.shape[0]):
        for j in range(W.shape[1]):
          old_v = W[i][j]
          W[i][j] = old_v + h
          fxh1 = f(W) # f(X+h)
          W[i][j] = old_v - h
          fxh2 = f(W) # f(X-h)
          dw[i][j] = (fxh1-fxh2)/(2 * h) # dx = (f(x+h)-f(x-h))/2h
          # dX = grad_numerical
          W[i][j] = old_v # reset
      return dw

    def _inner_loss(W):
      """
      Input: 
      Xi: A numpy array with shape (N，D)
      W: A numpy array with shape (D,C)
      """
      scores = X.dot(W) # (1, D) @ (D,C)
      # compute the probabilities in numerically stable way
      scores -= np.max(scores, axis=1,keepdims=True) # (N，C) 
      p = np.exp(scores) # (N,C) 
      p /= p.sum(axis=1,keepdims=True)  # normalize (N,C) 
      logp = np.log(p) # (N,C) 
      loss = -logp[y]  # negative log probability is the loss, shape (N,)
      
      # normalized hinge loss plus regularization
      loss = np.mean(loss) + reg * np.sum(W*W)
      return loss
    # 可以根据 y[i] 的值逆向求 loss
    # 设 y[i] = c, 那么 X[i] * W[:, [c]] 这一项有效，但是还是要求 X[i] * W, 因为要求
    # p /= p.sum()
    # f = lambda w: _inner_loss(w)
    # dW = _numeric_gradient(f)
    # print("dW shape2", dW.shape)
    # loss = _inner_loss(W)
    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # derivative = (loss(X+h) - loss(X-h)) / 2h
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        exp_scores = np.exp(scores)
        probs = exp_scores / np.sum(exp_scores)  # normalize

        # loss: negative log probability of correct class
        loss -= np.log(probs[y[i]])

        # gradient: dL/dW_{d,j} = X_{i,d} * (p_j - 1{j == y_i})
        # naive loop over all classes and dimensions
        for j in range(num_classes):
            for d in range(num_dims):
                if j == y[i]:
                    # correct class: gradient is X[i,d] * (p_j - 1)
                    dW[d, j] += X[i, d] * (probs[j] - 1)
                else:
                    # incorrect class: gradient is X[i,d] * p_j
                    dW[d, j] += X[i, d] * probs[j]

    # average over all training samples and add regularization
    loss = loss / num_train + reg * np.sum(W * W)
    dW = dW / num_train + 2 * reg * W


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.
    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength
    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    num_train = X.shape[0]

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the softmax loss, storing the           #
    # result in loss.                                                           #
    #############################################################################
    # Compute scores: (N, D) x (D, C) = (N, C)
    scores = X.dot(W)

    # Numerical stability: subtract max from each row
    scores -= np.max(scores, axis=1, keepdims=True)

    # Compute softmax probabilities: (N, C)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # Compute loss: sum of -log(prob of correct class) for all samples
    correct_log_probs = -np.log(probs[np.arange(num_train), y])
    loss = np.sum(correct_log_probs) / num_train + reg * np.sum(W * W)
    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the softmax            #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # Gradient: dL/dW = X^T * (probs - one_hot(y))
    # First, compute (probs - one_hot(y)): subtract 1 from correct class probs
    dscores = probs.copy()  # (N, C)
    dscores[np.arange(num_train), y] -= 1  # subtract 1 at correct class positions

    # dW = X^T * dscores: (D, N) x (N, C) = (D, C)
    dW = X.T.dot(dscores) / num_train + 2 * reg * W
    return loss, dW
