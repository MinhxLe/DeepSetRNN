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
    
        self.hidden_dim = hidden_dims[0]

        self.lstm = nn.LSTM(input_dim, hidden_dims[0], num_layers=n_layers, dropout=dropout)
        self.hidden1 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.output = nn.Linear(hidden_dims[1], n_class)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, sequence):
        sequence = utils.to_tensor(sequence)
        hidden = self.init_hidden()
        outputs, hidden = self.lstm(sequence.view(len(sequence), 1, -1), hidden)
        hidden1 = F.relu(self.dropout_layer(self.hidden1(outputs[-1])))
        return self.output(hidden1)

    def init_hidden(self):
            # Before we've done anything, we dont have any hidden state.
            # Refer to the Pytorch documentation to see exactly
            # why they have this dimensionality.
            # The axes semantics are (num_layers, minibatch_size, hidden_dim)
            return (torch.zeros(1, 1, self.hidden_dim),
                    torch.zeros(1, 1, self.hidden_dim))
