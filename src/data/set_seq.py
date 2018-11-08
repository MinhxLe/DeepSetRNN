import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

from src import utils

import sys
import logging

_LOGGER = logging.getLogger("set_seq")

def one_hot(element_idxs, n_class):
    x = np.zeros(n_class, element_idxs)

    x = torch.zeros(n_class).float()
    x[element_idx] = 1.0
    return x


def generate_embedding(
        pair_data,
        n_class,
        embedding_dims = 5,
        learning_rate = 0.001,
        n_epoch=100):
    """
    """

    #this is stochastic gradient descent
    
    W1 = Variable(torch.randn(embedding_dims,n_class).float(), requires_grad=True)
    W2 = Variable(torch.randn(n_class, embedding_dims).float(), requires_grad=True)
        
    for epoch in range(n_epoch):
        total_loss = 0
        for data, target in pair_data:
            y_true = torch.tensor([target]).long()

            z1 = W1[:,data]
            z2 = torch.matmul(W2, z1)
            
            log_softmax = F.log_softmax(z2, dim=0)

            loss = F.nll_loss(log_softmax.view(1,-1), y_true)
            total_loss += loss.data

            loss.backward()
            W1.data -= learning_rate*W1.grad.data 
            W2.data -= learning_rate*W2.grad.data

            W1.grad.data.zero_()
            W2.grad.data.zero_()
        
        _LOGGER.info("Loss at epo {}: {}".format(epoch, loss/len(pair_data)))

    return W1, W2


