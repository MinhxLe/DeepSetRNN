#!/usr/bin/env python
# coding: utf-8

# In[212]:

import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import os
import re
import string
import pickle
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import os
import importlib

from src import convert_dot_format
from src import experiment_utils


# DATA IMPUTATION

_ROOT_DATA_PATH = 'data/MIMIC3database/'
_RAW_PERSON_DATA_PATH = os.path.join(_ROOT_DATA_PATH, 'MIMIC3EachPerson')
_PROCESSED_DATA_PATH = os.path.join(_ROOT_DATA_PATH, 'processed')
_PROCESSED_PERSON_DATA_PATH = os.path.join(_PROCESSED_DATA_PATH, 'MIMIC3EachPerson')

data_sets = ['train', 'test']


for data_set in data_sets:
    individuals_path = os.path.join(_PROCESSED_PERSON_DATA_PATH, '{}_individuals.txt'.format(data_set))
    individuals_df = pd.read_csv(individuals_path)
    pattern = re.compile('^episode[0-9]+.csv$')
    append_header = True
    
    with open(os.path.join(_PROCESSED_PERSON_DATA_PATH, '{}_labels.csv'.format(data_set)), 'w') as main_csv_file:
        for individual in individuals_df['SUBJECT_ID']:
            individual = str(individual)
            raw_individual_root_path = os.path.join(_RAW_PERSON_DATA_PATH, data_set, str(individual))
            processed_individual_root_path = os.path.join(_PROCESSED_PERSON_DATA_PATH, data_set, str(individual))
            if not os.path.isdir(processed_individual_root_path):
                os.mkdir(processed_individual_root_path)
            admission_num = 1
            for file in os.listdir(raw_individual_root_path):
                if pattern.match(file):
                    df = pd.read_csv(os.path.join(raw_individual_root_path, file))
                    
                    df['SUBJECT_ID'] = individual
                    df['ADMISSION_NUM'] = admission_num  
                    admission_num += 1
                    df.to_csv(main_csv_file, header=append_header)
                    append_header = False
        main_csv_file.flush()
