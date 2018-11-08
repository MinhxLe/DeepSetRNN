import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SetSequenceModel(nn.Module):
    def __init__(self,\
            hidden_dim,
            n_class,
            embedding=None,
            embedding_dim=None,
            learn_embedding=False,
            freeze_embedding=True):
        super(SetSequenceModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        #getting embbedding
        if not embedding is None:
            self.embedding = nn.Embedding.from_pretrained(embedding, freeze=freeze_embedding)
            embedding_dim = embedding.size()[1]
        else:
            self.embedding = nn.Embedding(n_class, embedding_dim)
        #TODO multipl layers, dropout
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        #prediction
        self.output = nn.Linear(hidden_dim, n_class)
        self.hidden = self.init_hidden()
    
    def forward(self, set_sequence):
        hidden = self.hidden
        embeddings = [torch.sum(self.embedding(torch.tensor(set)),dim=0) for set in set_sequence]
        embeddings = torch.stack(embeddings).view(len(set_sequence), 1, -1)
        outputs, hidden = self.lstm(embeddings, hidden)
        return self.output(outputs.view(len(set_sequence),-1))

    def init_hidden(self):
            # Before we've done anything, we dont have any hidden state.
            # Refer to the Pytorch documentation to see exactly
            # why they have this dimensionality.
            # The axes semantics are (num_layers, minibatch_size, hidden_dim)
            return (torch.zeros(2, 1, self.hidden_dim),
                    torch.zeros(2, 1, self.hidden_dim))

