#!/usr/bin/env python
# coding: utf-8

# In[28]:


from src.data import school_contact, set_seq, math_tags
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import os
import importlib


# In[21]:


logging.info("starting logger")
_LOGGER = logging.getLogger('seq_model')
_LOGGER.setLevel(logging.DEBUG)


# In[33]:


data_name = 'school_contact'
data_module = importlib.import_module('src.data.{}'.format(data_name))
full_seq_path = os.path.join('data/raw/', data_module.DIR_NAME, data_module.FNAME)
sequences = set_seq.get_sequences(full_seq_path)


# In[34]:


#pretraining embedding
pairs = []
for sequence in sequences:
    for seq_set in sequence:
        for i in range(len(seq_set)):
            for j in range(i+1, len(seq_set)):
                pairs.append((seq_set[i],seq_set[j]))
EMBEDDING_DIM = 5 
W1,W2 = set_seq.generate_embedding(pairs, school_contact.N_ELEMENTS, embedding_dims=EMBEDDING_DIM, n_epoch=10)
W = W1.t().data+W2.data
W_norm = W.div(torch.norm(W,dim=1).view(-1,1))
torch.save(W_norm, 'data/processed/{}_embedding_normalized_d{}.pt'.format(data_name, EMBEDDING_DIM))


# In[46]:


#getting 1 hot representation of sequences
#for each sequence with length N, you will have a corresponding tensor with shaep N by N_ELEMENTS representing set of sequences
one_hot_sequences = []
for sequence in sequences:
    one_hot_sequences.append(torch.cat([torch.zeros(1, data_module.N_ELEMENTS, dtype=torch.float).     scatter(1, torch.LongTensor(elm_set).view(1,-1), 1) for elm_set in sequence]))


# In[56]:


from src.model.set_sequence import SetSequenceModel

embedding = W_norm
HIDDEN_DIM = 100
EMBEDDING_DIM = 5
model = SetSequenceModel(hidden_dim=HIDDEN_DIM,
                         n_class=data_module.N_ELEMENTS,
                         embedding=embedding)


# In[57]:


n_epoch = 50
n_seq = 100

split = int(n_seq*0.8)
train_sequences = sequences[:split]
train_targets =  one_hot_sequences[:split]

test_sequences = sequences[split:n_seq]
test_targets = one_hot_sequences[split:n_seq]

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=.9)


# In[58]:


test_losses = []
for sequence, target in zip(test_sequences, test_targets):
    model.hidden = model.init_hidden()
    logits = model(sequence)
    loss = loss_fn(logits[1:].view(-1),target[1:].view(-1))
    test_losses.append(loss.data)
_LOGGER.info("Validation Loss: {}".format(np.mean(test_losses)))


# In[ ]:


optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=.9)
losses = []

for epoch in range(n_epoch):
    curr_losses = []
    for sequence, target in zip(train_sequences,
                                train_targets):
        model.zero_grad()
        model.hidden = model.init_hidden()
        
        logits = model(sequence)
        loss = loss_fn(logits[1:].view(-1),target[1:].view(-1))
        curr_losses.append(loss.data)
        loss.backward()
        optimizer.step()
    mean_loss = np.mean(curr_losses)
    losses.append(mean_loss)
    _LOGGER.debug("epoch {}: {}".format(epoch, mean_loss))


# In[64]:


test_losses = []
for sequence, target in zip(test_sequences, test_targets):
    model.hidden = model.init_hidden()
    logits = model(sequence)
    loss = loss_fn(logits[1:].view(-1),target[1:].view(-1))
    test_losses.append(loss.data)
_LOGGER.info("Validation Loss: {}".format(np.mean(test_losses)))


# In[68]:


i = 10
logits = model(test_sequences[i])
prediction = torch.sigmoid(model(test_sequences[i]))


# In[69]:


print(test_sequences[i])


# In[70]:


np.argsort(prediction.data.numpy(),axis=1)[:,:1650:-1]


# In[52]:


np.max(prediction.data.numpy())


# In[ ]:




