import torch
import torch.nn as nn
import torch.nn.functional as F

class OneHotLogRegClassifier(nn.Module):
    def __init__(self,
            input_dim,
            n_class):
        
        self.n_class = n_class
        super(OneHotLogRegClassifier, self).__init__()
        self.linear = nn.Linear(input_dim, n_class)
    
    def forward(self, sequence):
        return self.linear(sequence).view(-1, self.n_class)

class OneHotFullyConnectedClassifier(nn.Module):
    def __init__(self,
        input_dim,
        n_class,
        hidden_dims=[100]
        ):
        
        super(OneHotFullyConnectedClassifier, self).__init__()
        self.n_class = n_class
       
        self.fc_layers = []

        assert(hidden_dims)
        if len(hidden_dims) == 1:
            self.fc_layers.append(nn.Linear(input_dim, n_class))
        else:
            self.fc_layers.append(nn.Linear(input_dim, hidden_dims[0]))
            for i in range(len(hidden_dims)-1):
                self.fc_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.fc_layers.append(nn.Linear(hidden_dims[-1], n_class))
        self.fc_layers = nn.ModuleList(self.fc_layers)
    def forward(self, sequence):
        output = self.fc_layers[0](sequence)
        
        if len(self.fc_layers) == 1:
            return output.view(-1, self.n_class)
        else:
            for layer in self.fc_layers[1:]:
                output = layer(F.relu(output))
        return output.view(-1, self.n_class)        

class OneHotLSTMClassifier(nn.Module):
    def __init__(self,
            input_dim,
            hidden_dims,
            n_layers,
            n_class,
            dropout=0.5):
        super(OneHotLSTMClassifier, self).__init__()
    
        self.hidden_dim = hidden_dims[0]

        self.lstm = nn.LSTM(input_dim, hidden_dims[0], num_layers=n_layers, dropout=dropout)
        self.hidden1 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.output = nn.Linear(hidden_dims[1], n_class)
        self.dropout_layer = nn.Dropout(p=dropout)

    def forward(self, sequence):
        hidden = self.init_hidden()
        outputs, hidden = self.lstm(sequence.view(len(sequence), 1, -1), hidden)
        hidden1 = F.relu(self.dropout_layer(self.hidden1(outputs.view(len(sequence), -1))))
        return self.output(hidden1)

    def init_hidden(self):
            # Before we've done anything, we dont have any hidden state.
            # Refer to the Pytorch documentation to see exactly
            # why they have this dimensionality.
            # The axes semantics are (num_layers, minibatch_size, hidden_dim)
            return (torch.zeros(1, 1, self.hidden_dim),
                    torch.zeros(1, 1, self.hidden_dim))
