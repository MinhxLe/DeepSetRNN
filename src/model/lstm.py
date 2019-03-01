import torch
import torch.nn as nn
import torch.nn.functional as F
from src import utils

class LSTMClassifier(nn.Module):
    def __init__(self,
            input_dim,
            n_class,
            hidden_dims,
            n_layers=1,
            dropout=0.5):
        super(LSTMClassifier, self).__init__()
    
        self.hidden_dims = hidden_dims

        self.lstm = nn.LSTM(input_dim, hidden_dims[0], num_layers=n_layers, dropout=dropout, batch_first=True)
        if len(hidden_dims) == 1:
            self.output = nn.Linear(hidden_dims[0], n_class)
        else: #len == 2
            self.hidden1 = nn.Linear(hidden_dims[0], hidden_dims[1])
            self.output = nn.Linear(hidden_dims[1], n_class)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, sequence):
        batch_size = sequence.shape[0]
        hidden = self.init_hidden(batch_size)
        outputs, hidden = self.lstm(sequence, hidden)
       
        if len(self.hidden_dims) == 1:
            return self.output(outputs[:,-1])
        else:
            hidden1 = F.relu(self.dropout_layer(self.hidden1(outputs[:,-1])))
            return self.output(hidden1)

    def init_hidden(self, batch_size):
            # Before we've done anything, we dont have any hidden state.
            # Refer to the Pytorch documentation to see exactly
            # why they have this dimensionality.
            # The axes semantics are (num_layers, minibatch_size, hidden_dim)
            return (torch.zeros(1, batch_size, self.hidden_dims[0]),
                    torch.zeros(1, batch_size, self.hidden_dims[0]))
