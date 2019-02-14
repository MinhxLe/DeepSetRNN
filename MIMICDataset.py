#!/usr/bin/env python
import pandas as pd
import os
DATA_PATH='data/MIMIC3database'


#procedures and diagnoses
diagnoses_path = os.path.join(DATA_PATH, 'DIAGNOSES_ICD.csv')
diagnoses = pd.read_csv(diagnoses_path,dtype={'ICD9_CODE': str})
procedures_path = os.path.join(DATA_PATH, 'PROCEDURES_ICD.csv')
procedures = pd.read_csv(procedures_path,dtype={'ICD9_CODE': str})

# embedding code found at
# https://github.com/clinicalml/embeddings/tree/master/eval

#icd9 to cui mapping for
def get_icd9_cui_mappings():
    cui_to_icd9 = {}
    icd9_to_cui = {}
    with open('external/embeddings/eval/cui_icd9.txt', 'r',  encoding="utf-8") as infile:
        data = infile.readlines()
        for row in data:
            ele = row.strip().split('|')
            if ele[11] == 'ICD9CM':
                cui = ele[0]
                icd9 = ele[10].replace('.','')
                if cui not in cui_to_icd9 and icd9 != '' and '-' not in icd9:
                    cui_to_icd9[cui] = icd9
                    icd9_to_cui[icd9] = cui
    return cui_to_icd9, icd9_to_cui

cui_to_icd9, icd9_to_cui = get_icd9_cui_mappings()


# In[87]:


missing_procedures = set([code for code in procedures['ICD9_CODE'] if not code in icd9_to_cui])



# In[45]:


embedding_path = 'external/embeddings/claims_codes_hs_300.txt.gz'
embeddings = pd.read_csv(embedding_path, compression='gzip', delimiter=' ', skiprows=[0], header=None)


# In[95]:


from gensim.models import KeyedVectors,Word2Vec
wv = KeyedVectors.load('data/MIMIC3database/process_icd9/icd9_all/processed_full.w2v')


# In[119]:


for word in wv.wv.vocab:
    if 'TICK' in word:
        print(word)


# In[ ]:




