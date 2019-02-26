#!/usr/bin/env python
# coding: utf-8

# In[178]:


import pandas as pd
import numpy as np
from collections import defaultdict
import os
import glob
import string
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str)

args = parser.parse_args()
data_set = args.data 


_ROOT_DATA_PATH = 'data/MIMIC3database/'
_RAW_PERSON_DATA_PATH = os.path.join(_ROOT_DATA_PATH, 'MIMIC3EachPerson')
_PROCESSED_DATA_PATH = os.path.join(_ROOT_DATA_PATH, 'processed')
_PROCESSED_PERSON_DATA_PATH = os.path.join(_PROCESSED_DATA_PATH, 'MIMIC3EachPerson')

individuals_path = os.path.join(_PROCESSED_DATA_PATH, 'MIMIC3EachPerson', 'test_individuals.txt'.format(data_set))
individuals_df = pd.read_csv(individuals_path)

features_of_interest = [
    'Hours',
    'Capillary refill rate',
    'Diastolic blood pressure',
    'Systolic blood pressure',
    'Fraction inspired oxygen',
    'Glascow coma scale total',
    'Respiratory rate',
    'Temperature',
    'Glucose',
    'Heart Rate',
    'Oxygen saturation',
    'pH']

def preprocess_time_series(raw_fname):
    df = pd.read_csv(raw_fname)
    df = df[features_of_interest]
    #df['Datetime']= pd.to_datetime(df['Hours']*1000000000*60*60)
    #df = df.set_index(pd.DatetimeIndex(df['Datetime']))
    #df.drop(['Hours', 'Datetime'], axis=1, inplace=True)
    df = df.fillna(df.mean().fillna(0))
    return df


individuals_path = os.path.join(_PROCESSED_PERSON_DATA_PATH, '{}_individuals.txt'.format(data_set))
individuals_df = pd.read_csv(individuals_path)

append_header = True
with open(os.path.join(_PROCESSED_PERSON_DATA_PATH, '{}_timeseries.csv'.format(data_set)), 'w') as main_csv_file:
    for individual in individuals_df['SUBJECT_ID']:
        individual = str(individual)
        raw_individual_root_path = os.path.join(_RAW_PERSON_DATA_PATH, data_set, str(individual))
        processed_individual_root_path = os.path.join(_PROCESSED_PERSON_DATA_PATH, data_set, str(individual))
        if not os.path.isdir(processed_individual_root_path):
            os.mkdir(processed_individual_root_path)
        admission_num = 1
        for file in os.listdir(raw_individual_root_path):
            if file.endswith("_timeseries.csv"):
                df = preprocess_time_series(os.path.join(raw_individual_root_path, file))
                
                df['SUBJECT_ID'] = individual
                df['ADMISSION_NUM'] = admission_num  
                admission_num += 1

                df.to_csv(os.path.join(processed_individual_root_path, file))
                df.to_csv(main_csv_file, header=append_header)
                append_header = False
        main_csv_file.flush()
