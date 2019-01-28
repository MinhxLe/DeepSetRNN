import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class SetSequenceModel(nn.Module):
    def __init__(self,
            hidden_dim,
            n_class,
            embedding=None,
            vocab_size=None,
            embedding_dim=None,
            freeze_embedding=False):
        super(SetSequenceModel, self).__init__()
        
        self.hidden_dim = hidden_dim[0]
        #getting embbedding
        if not embedding is None:
            self.embedding = nn.Embedding.from_pretrained(embedding, freeze=freeze_embedding)
            embedding_dim = embedding.size()[1]
        else:
            assert(vocab_size is not None and embedding_dim is not None)
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
        #TODO multipl layers, dropout
        #*2 for concatination of diagnoses and embedding
        self.lstm = nn.LSTM(embedding_dim*2, hidden_dim[0]) 
        #prediction
        self.hidden1 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.output = nn.Linear(hidden_dim[1], n_class)
        self.hidden = self.init_hidden()
    
    def forward(self, set_sequence):
        hidden = self.hidden
        
        diagnoses_sequence, procedures_sequence = zip(*set_sequence)
       

        #looking up embedding for each index
        #for every time step, for every set for every word
        
        diagnoses_embedding = [self.embedding(torch.tensor(set)) for set in diagnoses_sequence]
        #icd9 embedding is mean over 
        
        diagnoses_embedding = [torch.mean(embedding_set, dim=1) for embedding_set in diagnoses_embedding]
        #each time step you take the sum for the set 
        #TODO potentially a nonlinear transformation first
        diagnoses_embedding = [torch.sum(embedding_set, dim=0) for embedding_set in diagnoses_embedding]

        procedures_embedding = [self.embedding(torch.tensor(set)) for set in procedures_sequence]
        procedures_embedding = [torch.mean(embedding_set, dim=1) for embedding_set in procedures_embedding]
        procedures_embedding = [torch.sum(embedding_set, dim=0) for embedding_set in procedures_embedding]
    

        assert(len(procedures_embedding) == len(procedures_embedding))
        #concatination of 
        final_embeddings = [torch.cat((d_embed, p_embed))\
                for d_embed, p_embed in zip(diagnoses_embedding, procedures_embedding)]

        #embeddings = [torch.sum(self.embedding(torch.tensor(set)),dim=0) for set in set_sequence]
        
        final_embeddings = torch.stack(final_embeddings).view(len(set_sequence), 1, -1)
        
        outputs, hidden = self.lstm(final_embeddings, hidden)
        
        hidden1 = F.relu(self.hidden1(outputs.view(len(set_sequence), -1)))
        return self.output(hidden1)

    def init_hidden(self):
            # Before we've done anything, we dont have any hidden state.
            # Refer to the Pytorch documentation to see exactly
            # why they have this dimensionality.
            # The axes semantics are (num_layers, minibatch_size, hidden_dim)
            return (torch.zeros(1, 1, self.hidden_dim),
                    torch.zeros(1, 1, self.hidden_dim))

