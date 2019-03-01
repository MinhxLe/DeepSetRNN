

# In[2]:


from collections import defaultdict, namedtuple
import os
import string
import logging
import importlib

import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src import experiment_utils, utils


# In[3]:


logging.info("starting logger")
_LOGGER = logging.getLogger('VisitTimeSeries')
_LOGGER.setLevel(logging.DEBUG)


# # Reading data

# In[4]:


_ROOT_DATA_PATH = 'data/MIMIC3database/'
_PROCESSED_DATA_PATH = os.path.join(_ROOT_DATA_PATH, 'processed/MIMIC3EachPerson')


# In[5]:


train_series_df = pd.read_csv(os.path.join(_PROCESSED_DATA_PATH, 'train_timeseries.csv'),
                              index_col=0)

train_labels_df = pd.read_csv(os.path.join(_PROCESSED_DATA_PATH, 'train_labels.csv'),
                              index_col=0)

train_labels_df.drop(['Icustay', 'Ethnicity', 'Gender', 'Age', 'Height', 'Weight',
       'Length of Stay', 'Mortality'], axis=1, inplace=True)


train_series_df = train_series_df.set_index(['SUBJECT_ID', 'ADMISSION_NUM'])
train_labels_df = train_labels_df.set_index(['SUBJECT_ID', 'ADMISSION_NUM'])


# In[6]:


test_series_df = pd.read_csv(os.path.join(_PROCESSED_DATA_PATH, 'test_timeseries.csv'),
                              index_col=0)

test_labels_df = pd.read_csv(os.path.join(_PROCESSED_DATA_PATH, 'test_labels.csv'),
                              index_col=0)

test_labels_df.drop(['Icustay', 'Ethnicity', 'Gender', 'Age', 'Height', 'Weight',
       'Length of Stay', 'Mortality'], axis=1, inplace=True)

test_series_df = test_series_df.set_index(['SUBJECT_ID', 'ADMISSION_NUM'])
test_labels_df = test_labels_df.set_index(['SUBJECT_ID', 'ADMISSION_NUM'])


# In[7]:


n_features = train_series_df.iloc[0].shape[0]
n_class = train_labels_df.iloc[0].shape[0]


# In[8]:


def group_by_individual(series_df):
    #TODO THIS MUTATES series_df
    series_df['combined'] = series_df.values.tolist()
    series_df['combined'] = series_df['combined'].apply(lambda x : np.array(x))
    
    temp_df = series_df[['combined']]
    temp_df = temp_df.groupby(['SUBJECT_ID', 'ADMISSION_NUM'])['combined'].apply(list).to_frame()
    
    temp_df['combined'] = temp_df['combined'].apply(lambda x : np.array(x))
    return temp_df


# In[9]:


train_series_df = group_by_individual(train_series_df)
test_series_df = group_by_individual(test_series_df)


# In[ ]:



# In[10]:


test_indices_sorted = np.load("{}/test_indices_sorted_by_len.npy".format(_PROCESSED_DATA_PATH))
train_indices_sorted = np.load("{}/train_indices_sorted_by_len.npy".format(_PROCESSED_DATA_PATH))

train_indices_sorted = list(map(tuple, train_indices_sorted))
test_indices_sorted = list(map(tuple, test_indices_sorted))


# In[11]:


train_series_df = train_series_df.loc[train_indices_sorted]
train_labels_df = train_labels_df.loc[train_indices_sorted]

test_series_df = test_series_df.loc[test_indices_sorted]
test_labels_df = test_labels_df.loc[test_indices_sorted]


# # Training Model

# In[12]:


_MODEL_LOG_ROOT_PATH = 'logs/MIMIC3/VisitTimeSeries'
_MODEL_ROOT_PATH = 'models/MIMIC3/VisitTimeSeries'


# In[13]:


from src.model.lstm import LSTMClassifier

ModelArgs = namedtuple('HospitalVisitLSTMClassifier', 
                      ['hidden_dims',
                       'n_epoch',
                       'lr',
                       'momentum',
                      'n_layers',
                      'dropout',
                      'batch_size']
                      )
args = ModelArgs(
    hidden_dims=[500,200],
    n_epoch = 5,
    lr = 0.1,
    n_layers=1,
    momentum = 0.9,
    dropout=0.5,
    batch_size=50
)

model_name = str(args)

model = LSTMClassifier(hidden_dims=args.hidden_dims,
                       input_dim=n_features,
                       n_class=n_class,
                       dropout=args.dropout)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
experiment_utils.setup_model_logger(_LOGGER, model_name, _MODEL_LOG_ROOT_PATH)


# In[14]:


test_series = test_series_df['combined'].values
test_labels = test_labels_df.values

test_series_batch_padded = []
test_labels_batch = []

for i in range(0, len(test_series), args.batch_size):
    batch_slice = slice(i, min(i+args.batch_size, len(test_series)))
    test_series_batch_padded.append(utils.to_tensor(utils.pad_sequences(test_series[batch_slice])))
    test_labels_batch.append(utils.to_tensor(test_labels[batch_slice]))


# In[15]:


train_series = train_series_df['combined'].values
train_labels = train_labels_df.values

train_series_batch_padded = []
train_labels_batch = []

for i in range(0, len(train_series), args.batch_size):
    batch_slice = slice(i, min(i+args.batch_size, len(train_series)))
    train_series_batch_padded.append(utils.to_tensor(utils.pad_sequences(train_series[batch_slice])))
    train_labels_batch.append(utils.to_tensor(train_labels[batch_slice]))


# In[16]:


model = model.eval()
count = 0
total_loss = 0


for i in range(len(test_series_batch_padded)):
    model.zero_grad()
    logit = model(test_series_batch_padded[i])
    total_loss += loss_fn(logit, test_labels_batch[i]).data.numpy()
    del logit
    
_LOGGER.info("Initial Validation Loss: {}".format(total_loss/len(test_series_batch_padded)))


# In[ ]:


model = model.train()
_LOGGER.info("Training model...")
training_losses = []
for epoch in range(args.n_epoch):
    total_loss = 0
    for curr_series, outputs in zip(train_series_batch_padded, train_labels_batch):
        model.zero_grad()
        logits = model(curr_series)
        loss = loss_fn(logits, outputs)
        loss.backward()
        total_loss += loss.data.numpy()
        optimizer.step()

        del loss
        del logits
        
    mean_loss = total_loss/len(train_series_batch_padded)
    _LOGGER.info("Epoch: {}, Loss: {}".format(epoch, mean_loss))
    training_losses.append(mean_loss)


# In[ ]:


torch.save(model, "{}/{}.pt".format(_MODEL_ROOT_PATH, model_name))


# In[ ]:


model = model.eval()
count = 0
total_loss = 0

for idx in test_indices:
    curr_series = test_series_df.xs(idx, level=[0,1])
    output = test_labels_df.xs(idx, level=[0,1])
    
    logit = model(utils.to_tensor(curr_series))
    total_loss += loss_fn(logit, utils.to_tensor(output))
total_loss = total_loss.data.numpy()

_LOGGER.info("Final Validation Loss: {}".format(total_loss/len(test_indices)))

