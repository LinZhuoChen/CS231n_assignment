import numpy as np
from random import shuffle

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
  N = y.shape[0]
  D, C = W.shape
  dS = np.zeros((N,C))
  

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  sorce = np.dot(X, W)
  sorce = sorce - np.max(sorce, axis=-1)[..., np.newaxis]
  sorce_softmaxed = np.exp(sorce) / (np.sum(np.exp(sorce), axis=-1)[..., np.newaxis])
  loss = np.sum(-np.log(sorce_softmaxed[range(N), y])) / N
  loss = loss + reg * np.sum(W * W)

  dS = sorce_softmaxed
  dS[range(N), y] = dS[range(N), y] - 1
  dW = np.dot(X.T, dS)
  dW = dW / N + 2 * reg * W

  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  N = y.shape[0]
  D, C = W.shape
  dS = np.zeros((N,C))

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  sorce = np.dot(X, W)
  sorce = sorce - np.max(sorce, axis=-1)[..., np.newaxis]
  sorce_softmaxed = np.exp(sorce) / (np.sum(np.exp(sorce), axis=-1)[..., np.newaxis])
  loss = np.sum(-np.log(sorce_softmaxed[range(N), y])) / N
  loss = loss + reg * np.sum(W * W)

  dS = sorce_softmaxed
  dS[range(N), y] = dS[range(N), y] - 1

  dS = dS / N
  dW = np.dot(X.T, dS)
  dW = dW  + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

