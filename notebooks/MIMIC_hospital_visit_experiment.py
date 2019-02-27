#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('cd', '~/Research/Sriram/DeepSetRNN')


# In[136]:


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


# In[21]:


logging.info("starting logger")
_LOGGER = logging.getLogger('VisitTimeSeries')
_LOGGER.setLevel(logging.DEBUG)


# # Reading data

# In[36]:


_ROOT_DATA_PATH = 'data/MIMIC3database/'
_PROCESSED_DATA_PATH = os.path.join(_ROOT_DATA_PATH, 'processed/MIMIC3EachPerson')


# In[148]:


test_count = 1000

train_series_df = pd.read_csv(os.path.join(_PROCESSED_DATA_PATH, 'train_timeseries.csv'),
                              index_col=0,
                             nrows=test_count)

train_labels_df = pd.read_csv(os.path.join(_PROCESSED_DATA_PATH, 'train_labels.csv'),
                              index_col=0,
                             nrows=test_count)

train_series_df.drop('Hours', axis=1, inplace=True)
train_labels_df.drop(['Icustay', 'Ethnicity', 'Gender', 'Age', 'Height', 
                     'Weight', 'Length of Stay', 'Mortality'], axis=1, inplace=True)

train_features = list(train_series_df.groupby(['SUBJECT_ID', 'ADMISSION_NUM']))
train_labels = list(train_labels_df.groupby(['SUBJECT_ID', 'ADMISSION_NUM']))

for feature in train_features:
    feature[1].drop(['SUBJECT_ID', 'ADMISSION_NUM'], axis=1, inplace=True)
for label in train_labels:
    label[1].drop(['SUBJECT_ID', 'ADMISSION_NUM'], axis=1, inplace=True)
                                  
train_features.sort(key=lambda x : x[0])
train_labels.sort(key=lambda x: x[0])

train_labels = train_labels[:len(train_features)]

train_features = list(map(lambda x : x[1].values, train_features))
train_labels = list(map(lambda x: x[1].values, train_labels))


# In[149]:


test_series_df = pd.read_csv(os.path.join(_PROCESSED_DATA_PATH, 'test_timeseries.csv'),
                              index_col=0,
                             nrows=test_count)

test_labels_df = pd.read_csv(os.path.join(_PROCESSED_DATA_PATH, 'test_labels.csv'),
                              index_col=0,
                             nrows=test_count)

test_series_df.drop('Hours', axis=1, inplace=True)
test_labels_df.drop(['Icustay', 'Ethnicity', 'Gender', 'Age', 'Height', 
                     'Weight', 'Length of Stay', 'Mortality'], axis=1, inplace=True)

test_features = list(test_series_df.groupby(['SUBJECT_ID', 'ADMISSION_NUM']))
test_labels = list(test_labels_df.groupby(['SUBJECT_ID', 'ADMISSION_NUM']))

for feature in test_features:
    feature[1].drop(['SUBJECT_ID', 'ADMISSION_NUM'], axis=1, inplace=True)
for label in test_labels:
    label[1].drop(['SUBJECT_ID', 'ADMISSION_NUM'], axis=1, inplace=True)
                                  
test_features.sort(key=lambda x : x[0])
test_labels.sort(key=lambda x: x[0])

test_labels = test_labels[:len(test_features)]

test_features = list(map(lambda x : x[1].values, test_features))
test_labels = list(map(lambda x: x[1].values, test_labels))


# In[106]:


n_features = train_features[0].shape[1]
n_class = train_labels[0].shape[1]


# # Training Model

# In[150]:


_MODEL_LOG_ROOT_PATH = 'logs/MIMIC3/VisitTimeSeries'
_MODEL_ROOT_PATH = 'models/MIMIC3/VisitTimeSeries'


# In[151]:


#global objects
training_loss_map = {}
model_map = {}


# In[152]:


def run_train_and_log_experiments(model_name, model, loss_fn, optimizer):
    experiment_utils.setup_model_logger(_LOGGER, model_name, _MODEL_LOG_ROOT_PATH)
    _LOGGER.info(model_name)

    #initial test loss
    test_losses = experiment_utils.evaluate_validation_loss(model, loss_fn, test_inputs, test_outputs)
    _LOGGER.info("Initial Validation Loss: {}".format(np.mean(test_losses)))

    #training model
    training_losses = experiment_utils.train_model(model, loss_fn, optimizer,
                                                  args.n_epoch, train_inputs, 
                                                   train_outputs, _LOGGER)

    #saving model
    torch.save(model, '{}/{}.pt'.format(_MODEL_ROOT_PATH, model_name))

    #final validation loss
    test_losses = experiment_utils.evaluate_validation_loss(model, loss_fn, test_inputs, test_outputs)
    _LOGGER.info("final validation Loss: {}".format(np.mean(test_losses)))

    #saving model in global map
    model_map[model_name] = model
    training_loss_map[model_name] = training_losses


# In[153]:


from src.model.lstm import LSTMClassifier

ModelArgs = namedtuple('HospitalVisitLSTMClassifier', 
                      ['hidden_dims',
                       'n_epoch',
                       'lr',
                       'momentum',
                      'n_layers',
                      'dropout']
                      )
args = ModelArgs(
    hidden_dims=[1000,100],
    n_epoch = 10,
    lr = 0.1,
    n_layers=1,
    momentum = 0.9,
    dropout=0.5,
)

model_name = str(args)

model = LSTMClassifier(hidden_dims=args.hidden_dims,
                       input_dim=n_features,
                       n_class=n_class,
                       dropout=args.dropout)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)


# In[155]:


experiment_utils.setup_model_logger(_LOGGER, model_name, _MODEL_LOG_ROOT_PATH)

test_losses = experiment_utils.evaluate_validation_loss_template(model, 
                                          loss_fn, 
                                          test_features,
                                         test_labels)
_LOGGER.info("Initial Validation Loss: {}".format(np.mean(test_losses)))

training_losses = experiment_utils.train_model_template(model, loss_fn, optimizer,
                                                       args.n_epoch, train_features,
                                                       train_labels, _LOGGER)
torch.save(model, "{}/{}.pt".format(_MODEL_ROOT_PATH/ model_name))

test_losses = experiment_utils.evaluate_validation_loss_template(model, 
                                          loss_fn, 
                                          test_features,
                                         test_labels)

_LOGGER.info("Final Validation Loss: {}".format(np.mean(test_losses)))


# In[ ]:




