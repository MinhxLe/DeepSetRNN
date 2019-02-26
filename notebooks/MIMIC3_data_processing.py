#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')
get_ipython().run_line_magic('cd', '~/Research/Sriram/DeepSetRNN')

import pandas as pd
import numpy as np
from collections import defaultdict
import os
import string
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import os
import importlib

from src import convert_dot_format

DATA_PATH='data/MIMIC3database'


# In[4]:


#these are soft links
_RAW_DATA_DIR = 'data/MIMIC3database/raw'
_PROCESSED_DATA_DIR = 'data/MIMIC3database/processed'


# In[6]:


#reading in and converting to dot diagnoses and procedure dictionary and preprocessing
d_diagnoses = convert_dot_format.convert_to_dot_format('{}/D_ICD_DIAGNOSES.csv'.format(_RAW_DATA_DIR)
                                         ,'ICD9_CODE','diagnoses')

d_procedures = convert_dot_format.convert_to_dot_format('{}/D_ICD_PROCEDURES.csv'.format(_RAW_DATA_DIR)
                                         ,'ICD9_CODE','procedure')

#preprocessing code description by stripping punctuation
def preprocess_sentences(sentence):
    sentence = sentence.strip()
    sentence = sentence.lower()
    sentence = sentence.translate(str.maketrans('','',string.punctuation))
    return sentence


d_diagnoses['LONG_TITLE'] = d_diagnoses['LONG_TITLE'].apply(preprocess_sentences)
d_procedures['LONG_TITLE'] = d_procedures['LONG_TITLE'].apply(preprocess_sentences)

d_diagnoses.to_csv('{}/D_ICD_DIAGNOSES_dot_format.csv'.format(_PROCESSED_DATA_DIR))
d_procedures.to_csv('{}/D_ICD_PROCEDURES_dot_format.csv'.format(_PROCESSED_DATA_DIR))


# In[7]:


#splitting sentence into list
d_diagnoses_sentence_idx = d_diagnoses['LONG_TITLE'].apply(lambda x : pd.Series(x.split(' ')))
d_procedures_sentence_idx = d_procedures['LONG_TITLE'].apply(lambda x : pd.Series(x.split(' ')))

d_diagnoses_sentence_idx.fillna('PAD')
d_procedures_sentence_idx.fillna('PAD')


# In[8]:


word_idx_map = {'PAD' : 0, 'UNKNOWN':1}
word_set = set(['PAD', 'UNKNOWN'])
for row in d_diagnoses_sentence_idx.values:
    for word in row:
        word_idx_map[word] = 0
        word_set.add(word)
for row in d_procedures_sentence_idx.values:
    for word in row:
        word_idx_map[word] = 0
        word_set.add(word)


# In[10]:


# getting word embedding and stripping all words that are not in dataset
word_embedding = pd.read_csv('data/embeddings/w2vModel1Gram9Jan2019.txt',
                             delimiter=' ', skiprows=[0], header=None)

mimic_word_embedding = word_embedding[word_embedding[0].isin(word_set)]

embed_dim = 300
mimic_word_embedding = pd.DataFrame([['PAD'] + [0.]*embed_dim, ['UNKNOWN'] + [0.]*embed_dim]).append(mimic_word_embedding)

mimic_word_embedding.to_csv('data/embeddings/processed/w2vModel1Gram9Jan2019_mimic_only.txt')


# In[23]:


words = mimic_word_embedding[0]

#TODO fix this efficiency by building a map that only is in words in dictionary
def get_word_idx(word):
    idx_list = np.where(words == word)[0]
    if len(idx_list) == 0:
        return 1 #index of unknown symbol
    else:
        return idx_list[0]
    
for word in word_idx_map:
    word_idx_map[word] = get_word_idx(word)


# In[25]:


#tokenizing the sentence to idx
d_procedures_sentence_idx_temp = d_procedures_sentence_idx.applymap(lambda word : word_idx_map.get(word,1))
del d_procedures_sentence_idx 
d_procedures_sentence_idx = d_procedures_sentence_idx_temp

    
d_diagnoses_sentence_idx_temp = d_diagnoses_sentence_idx.applymap(lambda word : word_idx_map.get(word,1))
del d_diagnoses_sentence_idx
d_diagnoses_sentence_idx = d_diagnoses_sentence_idx_temp

d_diagnoses_sentence_idx.insert(0, 'ICD9_CODE', d_diagnoses['ICD9_CODE'])
d_procedures_sentence_idx.insert(0, 'ICD9_CODE', d_procedures['ICD9_CODE'])

d_diagnoses_sentence_idx.to_csv('data/MIMIC3database/processed/ICD9_diagnoses_sentences_mimic_idx.csv')
d_procedures_sentence_idx.to_csv('data/MIMIC3database/processed/ICD9_procedures_sentences_mimic_idx.csv')


# In[5]:


diagnoses_df = convert_dot_format.convert_to_dot_format('data/MIMIC3database/raw/DIAGNOSES_ICD.csv'
                                         ,'ICD9_CODE','diagnoses')
procedures_df = convert_dot_format.convert_to_dot_format('data/MIMIC3database/raw/PROCEDURES_ICD.csv'
                                         ,'ICD9_CODE','procedures')

diagnoses_df.drop(diagnoses_df.columns[0], axis=1, inplace=True)
procedures_df.drop(procedures_df.columns[0], axis=1,inplace=True)

admissions = pd.read_csv('data/MIMIC3database/raw/ADMISSIONS.csv')[['HADM_ID', 'ADMITTIME']]
diagnoses_df = diagnoses_df.merge(admissions, on='HADM_ID')
procedures_df = procedures_df.merge(admissions, on='HADM_ID')

diagnoses_df.to_csv('data/MIMIC3database/processed/DIAGNOSES_ICD_dot_format.csv')
procedures_df.to_csv('data/MIMIC3database/processed/PROCEDURES_ICD_dot_format.csv')


# In[32]:


d_diagnoses = pd.read_csv('data/MIMIC3database/processed/ICD9_diagnoses_sentences_mimic_idx.csv', index_col=0)
d_procedures = pd.read_csv('data/MIMIC3database/processed/ICD9_procedures_sentences_mimic_idx.csv', index_col=0)

diagnoses_df = pd.read_csv('data/MIMIC3database/processed/DIAGNOSES_ICD_dot_format.csv', index_col=0)
procedures_df = pd.read_csv('data/MIMIC3database/processed/PROCEDURES_ICD_dot_format.csv', index_col=0)


# In[29]:


diagnoses_df_temp = diagnoses_df.merge(d_diagnoses, on='ICD9_CODE')
del diagnoses_df
diagnoses_df = diagnoses_df_temp

procedures_df_temp = procedures_df.merge(d_procedures, on='ICD9_CODE')
del procedures_df
procedures_df = procedures_df_temp


# In[46]:


diagnoses_procedures_df = pd.merge(diagnoses_df, procedures_df, on=['SUBJECT_ID', 'HADM_ID', 'ADMITTIME'], 
         how='outer', suffixes=('_diagnoses', '_procedures'))


# In[57]:


diagnoses_procedures_df.fillna(0, inplace=True) #WRONG since you fill other things too

#because of original nans you need to cast type back to int64
diagnoses_procedures_df[['ROW_ID_procedures', 'SEQ_NUM_procedures','0_procedures', '1_procedures', '2_procedures', '3_procedures',
       '4_procedures', '5_procedures', '6_procedures', '7_procedures',
       '8_procedures', '9_procedures', '10_procedures', '11_procedures',
       '12_procedures', '13_procedures', '14_procedures', '15_procedures',
       '16_procedures', '17_procedures', '18_procedures', '19_procedures',
       '20_procedures', '21_procedures']] = diagnoses_procedures_df[['ROW_ID_procedures', 'SEQ_NUM_procedures','0_procedures', '1_procedures', '2_procedures', '3_procedures',
       '4_procedures', '5_procedures', '6_procedures', '7_procedures',
       '8_procedures', '9_procedures', '10_procedures', '11_procedures',
       '12_procedures', '13_procedures', '14_procedures', '15_procedures',
       '16_procedures', '17_procedures', '18_procedures', '19_procedures',
       '20_procedures', '21_procedures']].astype('int64', inplace=True)


# In[59]:


diagnoses_procedures_df['DIAGNOSES_SENTENCES'] = diagnoses_procedures_df[['0_diagnoses', '1_diagnoses',
       '2_diagnoses', '3_diagnoses', '4_diagnoses', '5_diagnoses',
       '6_diagnoses', '7_diagnoses', '8_diagnoses', '9_diagnoses',
       '10_diagnoses', '11_diagnoses', '12_diagnoses', '13_diagnoses',
       '14_diagnoses', '15_diagnoses', '16_diagnoses', '17_diagnoses',
       '18_diagnoses', '19_diagnoses', '20_diagnoses', '21_diagnoses', '22',
       '23', '24', '25', '26', '27', '28', '29', '30', '31']].values.tolist()

diagnoses_procedures_df['PROCEDURES_SENTENCES'] = diagnoses_procedures_df[
    ['0_procedures', '1_procedures', '2_procedures', '3_procedures',
       '4_procedures', '5_procedures', '6_procedures', '7_procedures',
       '8_procedures', '9_procedures', '10_procedures', '11_procedures',
       '12_procedures', '13_procedures', '14_procedures', '15_procedures',
       '16_procedures', '17_procedures', '18_procedures', '19_procedures',
       '20_procedures', '21_procedures']].values.tolist()


# In[60]:


diagnoses_procedures_df.sort_values(by=['SUBJECT_ID', 'ADMITTIME', 'SEQ_NUM_diagnoses'], inplace=True)


# In[1]:


diagnoses_procedures_df.to_csv('data/MIMIC3database/processed/ICD9_diagnoses_procedures_mimic_idx_sentences_sorted.csv')


# In[ ]:


#only retaining top_n popular disease
top_n = 100

diagnoses_procedures_df = pd.read_csv('data/MIMIC3database/processed/ICD9_diagnoses_procedures_mimic_idx_sentences_sorted.csv', index_col=0)
diagnoses_counts = diagnoses_procedures_df['ICD9_CODE_diagnoses'].value_counts()
procedures_counts = diagnoses_procedures_df['ICD9_CODE_procedures'].value_counts()

diagnoses_set = set(diagnoses_counts.keys()[:top_n])
procedures_set = set(procedures_counts.keys()[:top_n])

diagnoses_procedures_df = diagnoses_procedures_df[diagnoses_procedures_df['ICD9_CODE_diagnoses'].isin(diagnoses_set) 
                                                        | diagnoses_procedures_df['ICD9_CODE_procedures'].isin(procedures_set)]
diagnoses_procedures_df.to_csv('data/MIMIC3database/processed/ICD9_diagnoses_procedures_mimic_idx_sentences_sorted_top_{}.csv'.format(top_n))


# In[5]:


diagnoses_procedures_df = pd.read_csv('data/MIMIC3database/processed/ICD9_diagnoses_procedures_mimic_idx_sentences_sorted.csv', index_col=0)

#converting sentences to a single column
diagnoses_procedures_df['DIAGNOSES_SENTENCES'] = diagnoses_procedures_df[['0_diagnoses', '1_diagnoses',
       '2_diagnoses', '3_diagnoses', '4_diagnoses', '5_diagnoses',
       '6_diagnoses', '7_diagnoses', '8_diagnoses', '9_diagnoses',
       '10_diagnoses', '11_diagnoses', '12_diagnoses', '13_diagnoses',
       '14_diagnoses', '15_diagnoses', '16_diagnoses', '17_diagnoses',
       '18_diagnoses', '19_diagnoses', '20_diagnoses', '21_diagnoses', '22',
       '23', '24', '25', '26', '27', '28', '29', '30', '31']].values.tolist()

diagnoses_procedures_df['PROCEDURES_SENTENCES'] = diagnoses_procedures_df[
    ['0_procedures', '1_procedures', '2_procedures', '3_procedures',
       '4_procedures', '5_procedures', '6_procedures', '7_procedures',
       '8_procedures', '9_procedures', '10_procedures', '11_procedures',
       '12_procedures', '13_procedures', '14_procedures', '15_procedures',
       '16_procedures', '17_procedures', '18_procedures', '19_procedures',
       '20_procedures', '21_procedures']].values.tolist()

diagnoses_procedures_df.drop(labels=['0_diagnoses', '1_diagnoses',
       '2_diagnoses', '3_diagnoses', '4_diagnoses', '5_diagnoses',
       '6_diagnoses', '7_diagnoses', '8_diagnoses', '9_diagnoses',
       '10_diagnoses', '11_diagnoses', '12_diagnoses', '13_diagnoses',
       '14_diagnoses', '15_diagnoses', '16_diagnoses', '17_diagnoses',
       '18_diagnoses', '19_diagnoses', '20_diagnoses', '21_diagnoses', '22',
       '23', '24', '25', '26', '27', '28', '29', '30', '31'], axis=1, inplace=True)

diagnoses_procedures_df.drop(labels=['0_procedures', '1_procedures', '2_procedures', '3_procedures',
       '4_procedures', '5_procedures', '6_procedures', '7_procedures',
       '8_procedures', '9_procedures', '10_procedures', '11_procedures',
       '12_procedures', '13_procedures', '14_procedures', '15_procedures',
       '16_procedures', '17_procedures', '18_procedures', '19_procedures',
       '20_procedures', '21_procedures'], axis=1, inplace=True)

diagnoses_procedures_df.to_csv("data/MIMIC3database/processed/ICD9_diagnoses_procedures_mimic_idx_sentences_top_100_sorted_concat.csv")


# In[9]:


diagnoses_procedures_df = pd.read_csv("data/MIMIC3database/processed/ICD9_diagnoses_procedures_mimic_idx_sentences_top_100_sorted_concat.csv", index_col=0)
top_n = 100 #should be the same as before


diagnoses_counts = diagnoses_procedures_df['ICD9_CODE_diagnoses'].value_counts()
procedures_counts = diagnoses_procedures_df['ICD9_CODE_procedures'].value_counts()

diagnoses_set = set(diagnoses_counts.keys()[:top_n])
procedures_set = set(procedures_counts.keys()[:top_n])

data = list(diagnoses_procedures_df.groupby(['SUBJECT_ID']))
data = [(subject_id, list(subject_data.groupby(['HADM_ID', 'ADMITTIME']))) for subject_id, subject_data in data]

inputs = []
for _, subject in data:
    series = []
    for _, timestep in subject:
        timestep = timestep[timestep['ICD9_CODE_diagnoses'].isin(diagnoses_set)
                           | timestep['ICD9_CODE_procedures'].isin(procedures_set)]
        if len(timestep) > 0:
            series.append((np.stack(timestep['DIAGNOSES_SENTENCES'],axis=0), np.stack(timestep['PROCEDURES_SENTENCES'],axis=0)))
        #for _, timestep in timesteps:
        #    print(timestep)
    if len(series) > 0:
        inputs.append(series)

diagnoses_idx_map = {}
for i, code in enumerate(diagnoses_counts.keys()[:top_n]):
    diagnoses_idx_map[code] = i

outputs = []

def get_onehot_vector(indices, top_n):
    prediction = np.zeros(top_n, dtype='float32')
    prediction[indices] = 1
    return prediction

for _, subject in data:
    series = []
    for _, timestep in subject:
        indices = [diagnoses_idx_map[key] for key in timestep['ICD9_CODE_diagnoses']                   if key in diagnoses_set]
        if len(timestep) > 0:
            series.append(get_onehot_vector(indices, top_n))
    if len(series) > 0:
        all_outputs.append(np.array(series))
#outputs = list(map(get_key, outputs))

pickle.dump(inputs, open('data/MIMIC3database/processed/ICD9_diagnoses_procedures_sequences_features.pkl', 'wb'))
pickle.dump(outputs, open('data/MIMIC3database/processed/ICD9_diagnoses_procedures_sequences_labels.pkl', 'wb'))

