import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import scipy.special as sp
import time
from scipy.optimize import minimize

import ce_data_generator as dg

def cross_entropy_softmax_loss(Wb, x, y, num_class, n, feat_dim):
    W = np.reshape(Wb[:-num_class], (num_class, feat_dim))
    b = np.reshape(Wb[-num_class:], (num_class, 1))

    scores = W @ x.T + b

    scores -= np.max(scores, axis=0, keepdims=True)
    exp_scores = np.exp(scores)
    probs = exp_scores / np.sum(exp_scores, axis=0, keepdims=True)

    correct_logprobs = -np.log(probs[y, np.arange(n)])
    loss = np.sum(correct_logprobs) / n

    return loss

def linear_classifier_test(Wb, x, y, num_class):
    n_test = x.shape[0]
    feat_dim = x.shape[1]
    
    Wb = np.reshape(Wb, (-1, 1))
    b = Wb[-num_class:].squeeze()
    W = np.reshape(Wb[:-num_class], (num_class, feat_dim))
    accuracy = 0

    s = x@W.T + b
    
    res = np.argmax(s, axis = 1)

    accuracy = (res == y).astype('uint8').sum()/n_test
    
    return accuracy

if __name__ == '__main__':
    num_class = 4

    sigma = 1.0

    print('number of classes: ',num_class,' sigma for data scatter:',sigma)
    if num_class == 4:
        n_train = 400
        n_test = 100
        feat_dim = 2
    else:  # then 3
        n_train = 300
        n_test = 60
        feat_dim = 2

    print('generating training data')
    x_train, y_train = dg.generate(number=n_train, seed=None, plot=True, num_class=num_class, sigma=sigma)

    print('generating test data')
    x_test, y_test = dg.generate(number=n_test, seed=None, plot=False, num_class=num_class, sigma=sigma)

    print('training softmax classifier...')
    w0 = np.random.normal(0, 1, (2 * num_class + num_class))
    result = minimize(cross_entropy_softmax_loss, w0, args=(x_train, y_train, num_class, n_train, feat_dim))

    print('testing softmax classifier...')

    Wb = result.x
    print('accuracy of softmax loss: ', linear_classifier_test(Wb, x_test, y_test, num_class)*100,'%')
