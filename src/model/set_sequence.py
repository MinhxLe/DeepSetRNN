import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SetSequenceModel:
    def __init__(self,\
            embedding_dim,
            hidden_dim,
            vocab_size,
            tagset_size,
            embedding=None,
            learn_embedding=False):

        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(embedding_size, hidden_dim)
        #parameters
        self.hidden = self._init_hidden():
    


    def _init_hidden(self):
        #c and h matrix
        return (torch.zeros(1,1, self.hidden_dim),\
                )

    def forward(self, set_sequence):
        pass
