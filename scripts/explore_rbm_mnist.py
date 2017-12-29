"""
script to explore the dynamics of rbm trianed on mnist dataset
"""


""" import """
import numpy as np
import matplotlib.pyplot as plt
from bm.rbm import BernoulliRBM
from bm.utils import (Stopwatch, im_plot, plot_confusion_matrix)
from bm.utils.dataset import load_mnist

""" load data, load model and get parameters """
X, y = load_mnist(mode='train', path='./data/')
X /= 255.
X_test, y_test = load_mnist(mode='test', path='./data/')
X_test /= 255.
print (X.shape, y.shape, X_test.shape, y_test.shape)

rbm = BernoulliRBM.load_model('./models/rbm_mnist/')

weights = rbm.get_tf_params(scope='weights')
W = weights['W']
vb = weights['vb']
hb = weights['hb']


def print_model_params(rbm):
    print('parameters of the model:')
    list_params = ['n_visible', 'n_hidden', 'n_gibbs_steps', 'batch_size', 'dropout', 'sparsity_target']
    for param in list_params:
        print('    {} = {}'.format(param, rbm.__getattribute__(param)))
    print('')
print_model_params(rbm)



temp = rbm.transform(X[:5])