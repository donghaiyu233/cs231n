import numpy as np
from random import shuffle
#from past.builtins import xrange

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in range(num_train):
        scores = np.dot(X[i], W)
        #scores:(1,C)
        correct_scores = scores[y[i]]
        exp_sum = np.sum(np.exp(scores))
        loss += np.log(exp_sum) - correct_scores
        dW[:,y[i]] += -X[i]
        
        for j in range(num_classes):
            dW[:,j] += (np.exp(scores[j]) / exp_sum) * X[i]
  loss /= num_train
  dW /= num_train
  #增加正则项
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X, W)
  num_train = X.shape[0]
  rows = range(num_train)
  correct_class_score = scores[rows,y]
  correct_class_score = np.reshape(correct_class_score,[num_train,1])
  exp_sum = np.sum(np.exp(scores), axis = 1).reshape(num_train,1)
  loss += np.sum(np.log(exp_sum) - correct_class_score)
  p = np.exp(scores) / exp_sum
  p[rows, y] += -1
  dW = X.T.dot(p)
  
  loss /= num_train
  dW /= num_train
  #正则项
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

