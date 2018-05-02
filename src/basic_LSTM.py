import numpy as np
import time
import torch

import torch.nn as nn
# Hyper Parameters
sequence_length = 28
input_size = 28
hidden_size = 128
num_layers = 2
num_classes = 10
batch_size = 100
num_epochs = 2
learning_rate = 0.01

class RNN(nn.Module):
    """
    many to one RNN model 
    """
    def __init__(self, input_size, hidden_size,num_layers,num_classes\
            ):
        pass
