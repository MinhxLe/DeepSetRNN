import torch
import torch.nn as nn
import torch.nn.functional as F

class EmbeddedLogRegClassifier(nn.Module):
    def __init__(self,
            n_class,
            embedding):
        self.n_class = n_class
        super(EmbeddedLogRegClassifier, self).__init__()
        
        self.embedding = nn.Embedding.from_pretrained(embedding, freeze=True)
        embedding_dim = embedding.size()[1]
        self.linear = nn.Linear(embedding_dim*2, n_class)
    
    def forward(self, set_sequence):
        diagnoses_sequence, procedures_sequence = zip(*set_sequence)
        
        diagnoses_embedding = [self.embedding(torch.tensor(set)) for set in diagnoses_sequence]
        diagnoses_embedding = [torch.mean(embedding_set, dim=1) for embedding_set in diagnoses_embedding]
        diagnoses_embedding = [torch.sum(embedding_set, dim=0) for embedding_set in diagnoses_embedding]

        procedures_embedding = [self.embedding(torch.tensor(set)) for set in procedures_sequence]
        procedures_embedding = [torch.mean(embedding_set, dim=1) for embedding_set in procedures_embedding]
        procedures_embedding = [torch.sum(embedding_set, dim=0) for embedding_set in procedures_embedding]

        final_embeddings = [torch.cat((d_embed, p_embed))\
                for d_embed, p_embed in zip(diagnoses_embedding, procedures_embedding)]
        final_embeddings = torch.stack(final_embeddings).view(len(set_sequence), 1, -1)

        return self.linear(final_embeddings).view(-1, self.n_class)
