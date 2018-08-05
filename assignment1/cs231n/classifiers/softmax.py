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
	num_train = X.shape[0]
	num_classes = W.shape[1]

	#############################################################################
	# TODO: Compute the softmax loss and its gradient using explicit loops.     #
	# Store the loss in loss and the gradient in dW. If you are not careful     #
	# here, it is easy to run into numeric instability. Don't forget the        #
	# regularization!                                                           #
	#############################################################################

	for i in range(num_train):
		scores = X[i].dot(W)
		scores_shift = scores - np.max(scores)
		scores_shift_exp = np.exp(scores_shift)
		ls = np.exp(scores_shift[y[i]]) / scores_shift_exp.sum()
		dy = np.exp(scores_shift) / scores_shift_exp.sum()
		dy[y[i]] -= 1
		loss += -np.log(ls)
		dy = np.expand_dims(dy, axis=0)
		dW += np.expand_dims(X[i], axis=1).dot(dy)
	
	loss /= num_train
	dW /= num_train

	loss += reg * (W * W).sum()
	dW += reg * 2 * W

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
	num_train = X.shape[0]
	num_classes = W.shape[1]

	#############################################################################
	# TODO: Compute the softmax loss and its gradient using no explicit loops.  #
	# Store the loss in loss and the gradient in dW. If you are not careful     #
	# here, it is easy to run into numeric instability. Don't forget the        #
	# regularization!                                                           #
	#############################################################################
	
	scores = X.dot(W)
	scores_max = scores.max(axis=1)
	scores_shift = scores - np.expand_dims(scores_max, axis=1)
	scores_shift_exp = np.exp(scores_shift)

	correct = scores_shift_exp[range(num_train), y.squeeze()]
	exp_sum = np.expand_dims(scores_shift_exp.sum(axis=1), axis=1)

	loss = np.expand_dims(correct, axis=1) / exp_sum
	loss = np.sum(-np.log(loss)) / num_train
	loss += reg * np.sum(W * W)

	dy = scores_shift_exp / exp_sum
	dy[range(num_train), y.squeeze()] += -1
	dW = X.T.dot(dy) / num_train
	dW += reg * 2 * W


	#############################################################################
	#                          END OF YOUR CODE                                 #
	#############################################################################

	return loss, dW

