#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import os, sys
import importlib

from src.data import school_contact, set_seq, math_tags
from src.model.set_sequence import SetSequenceModel

_LOGGER = logging.getLogger('seq_model')
_LOGGER.setLevel(logging.INFO)

_LOGGER.addHandler(logging.FileHandler('logs/math_tags_seq_model.txt'))
_LOGGER.addHandler(logging.StreamHandler(sys.stdout))

data_name = 'school_contact'
_LOGGER.info("running seq model on {}".format(data_name))
data_module = importlib.import_module('src.data.{}'.format(data_name))
full_seq_path = os.path.join('data/raw/', data_module.DIR_NAME, data_module.FNAME)
sequences = set_seq.get_sequences(full_seq_path)


#pretraining embeddings
_LOGGER.info("generating embedding")
EMBEDDING_DIM = 5 
pairs = []
for sequence in sequences:
    for seq_set in sequence:
        for i in range(len(seq_set)):
            for j in range(i+1, len(seq_set)):
                pairs.append((seq_set[i],seq_set[j]))
W1,W2 = set_seq.generate_embedding(pairs, school_contact.N_ELEMENTS, embedding_dims=EMBEDDING_DIM, n_epoch=10)
W = W1.t().data+W2.data
W_norm = W.div(torch.norm(W,dim=1).view(-1,1))
torch.save(W_norm, 'data/processed/{}_embedding_normalized_d{}.pt'.format(data_name, EMBEDDING_DIM))

#getting 1 hot representation of sequences
#for each sequence with length N, you will have a corresponding tensor with shaep N by N_ELEMENTS representing set of sequences
one_hot_sequences = []
for sequence in sequences:
    one_hot_sequences.append(torch.cat([torch.zeros(1, data_module.N_ELEMENTS, dtype=torch.float).     scatter(1, torch.LongTensor(elm_set).view(1,-1), 1) for elm_set in sequence]))

HIDDEN_DIM = 100
model = SetSequenceModel(hidden_dim=HIDDEN_DIM,
                         n_class=data_module.N_ELEMENTS,
                         embedding=W_norm)

#Training
n_epoch = 20
n_seq = len(sequences)

split = int(n_seq*0.8)
train_sequences = sequences[:split]
train_targets =  one_hot_sequences[:split]

test_sequences = sequences[split:n_seq]
test_targets = one_hot_sequences[split:n_seq]

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=.9)


test_losses = []
for sequence, target in zip(test_sequences, test_targets):
    model.hidden = model.init_hidden()
    logits = model(sequence)
    loss = loss_fn(logits[1:].view(-1),target[1:].view(-1))
    test_losses.append(loss.data)
_LOGGER.info("Initial Validation Loss: {}".format(np.mean(test_losses)))
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
    _LOGGER.info("epoch {}: {}".format(epoch, mean_loss))

test_losses = []
for sequence, target in zip(test_sequences, test_targets):
    model.hidden = model.init_hidden()
    logits = model(sequence)
    loss = loss_fn(logits[1:].view(-1),target[1:].view(-1))
    test_losses.append(loss.data)
_LOGGER.info("Final Validation Loss: {}".format(np.mean(test_losses)))


#saving embedding and model
torch.save(model.state_dict(), "model/{}_set_seq.mdl".format(data_name))
