

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string, re, sys, pickle 
import pandas as pd 
from nltk.tokenize import RegexpTokenizer
#retain only alphanumeric
tokenizer = RegexpTokenizer(r'\w+')


def convert_procedure_to_dot_format (k): 
  if len(k) == 4: 
    k = k[0:2]+'.'+k[2:len(k)] ## rename 
  elif len(k) == 5:
    k = k[0:3]+'.'+k[3:len(k)] ## rename  
  elif len(k) == 3: ## len is 3, we use the 1st 2 digits
    k = k[0:2]+'.'+k[2:len(k)] ## rename  
  else: 
    print ('incorrect format? '+k)
  return (k)

def convert_diagnosis_to_dot_format (k):
  if re.match("^E",k): ## E code has Exxx.y format ?? why ?
    if len(k) == 5: 
      k = k[0:4]+'.'+k[4:len(k)] ## rename 
    # can have length 4
    return (k)
  if len(k) == 4: 
    k = k[0:3]+'.'+k[3:len(k)] ## rename 
  elif len(k) == 5:
    k = k[0:3]+'.'+k[3:len(k)] ## rename  
  elif len(k) == 3: ## do nothing ? 
    return (k)
  else: 
    print ('incorrect format? '+k)
  return (k)


## 
def convert_to_dot_format (filename,colname,datatype='procedure'): 
	df = pd.read_csv( filename, dtype=str )
	df = df.dropna()
	#print (df.loc[0:10,:])
	row_iterator = df.iterrows() ## is it faster ?? 
	new_name = []
	for i, row in row_iterator:
		if datatype=='procedure': 
			new_name.append ( convert_procedure_to_dot_format ( row[colname] ) )
		else: 
			new_name.append ( convert_diagnosis_to_dot_format ( row[colname] ) )
	# new df 
	df [colname] = new_name 
	#print (df.loc[0:10,:])
	return df 


# procedure = convert_to_dot_format ( '/u/flashscratch/d/datduong/MIMIC3database/PROCEDURES_ICD.csv','ICD9_CODE','procedure') 
# diagnosis = convert_to_dot_format ( '/u/flashscratch/d/datduong/MIMIC3database/DIAGNOSES_ICD.csv','ICD9_CODE','diagnosis')

# person_full_icd = pd.concat ( [diagnosis, procedure], ignore_index=True )
# print ( person_full_icd.shape )

# person_full_icd = person_full_icd.sort_values(by=["SUBJECT_ID","HADM_ID","SEQ_NUM"])

# person_full_icd.to_csv( '/u/flashscratch/d/datduong/MIMIC3database/DIAGNOSES_PROCEDURES_ICD_dot_format.csv', index=False )

# print ( person_full_icd.loc[0:10,:] ) 


## must fix labels so we use leaf nodes 

def Punctuation(string): 
  # punctuation marks 
  punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
  # traverse the given string and if any punctuation 
  # marks occur replace it with null 
  for x in string.lower(): 
    if x in punctuations: 
      string = string.replace(x, " ") 
  # Print string without punctuation 
  return string


## parents/children look up index 

# parents = pickle.load ( open("/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019/parent_icd.pickle","rb") ) 
# children = pickle.load ( open("/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019/children_icd.pickle","rb") ) 

# Preferred_Label = pickle.load ( open("/u/flashscratch/d/datduong/MIMIC3database/format10Jan2019/Preferred_Label.pickle","rb") ) 

# ## it is only sensible to make prediction based on leaf nodes (most specific and bill-able) ??

# is_children = 0 
# codes = set ( list ( person_full_icd['ICD9_CODE'] ) ) 
# for k,val in enumerate(codes) : 
  # if (val in parents) and (val not in children) : ## leaf has parent but no children. actually, can just use @children, but let's be careful 
    # is_children = is_children + 1 

# print ('\nbefore fixing to lower node, fraction of children node over total unique code used {}'.format ( is_children*1.0 / len(codes)))


# print ('\n\nwe will convert higher level icd into leaf node ... if possible\n')

def convert_to_leaf (node,children,Preferred_Label): ## must convert to all leaf nodes 
  ch = children[node] ## all children of this node 
  for c in ch: 
    if Preferred_Label[c] == Preferred_Label[node] : ## exact match. 
      return c 
    #
    # c_def = Punctuation(Preferred_Label[c]).split()
    # node_def = Punctuation(Preferred_Label[node]).split() 
    # if len ( set(node_def).intersection(c_def) ) == len(node_def):
    #   return c 
    # if len ( set(node_def).intersection(c_def) ) == len(c_def):
    #   return c 
  return node ## worst case ?? 

# codes = set ( list ( person_full_icd['ICD9_CODE'] )) ## create a mapping 
# map_to_leaf = {}
# for j in codes: 
  # if (j in parents) and (j in children) : ## leaf will not have children  
    # new_j = convert_to_leaf (j,children,Preferred_Label)
    # if j == new_j: continue ## can not reduce the parent node to a lower level node  
    # map_to_leaf[j] = new_j 

# pickle.dump (map_to_leaf, open("/u/flashscratch/d/datduong/MIMIC3database/map_to_leaf.pickle","wb"))


# codes = list ( person_full_icd['ICD9_CODE'] ) ## convert. 
# for k,val in enumerate(codes): 
  # if val in map_to_leaf: 
    # codes[k] = map_to_leaf[val]


# person_full_icd['ICD9_CODE'] = codes ## should all be in leaves (or at least as much as possible)


# person_full_icd.to_csv( '/u/flashscratch/d/datduong/MIMIC3database/DIAGNOSES_PROCEDURES_ICD_dot_format_to_leaf.csv', index=False )

# ## do some tally count 
# is_children = 0 
# codes = set ( list ( person_full_icd['ICD9_CODE'] ) ) 
# for k,val in enumerate(codes) : 
  # if (val in parents) and (val not in children) : 
    # is_children = is_children + 1 

# print ('\nfraction of children node used over total unique code used {}'.format ( is_children*1.0 / len(codes)))


# ## now we look at all occ. (can see same icd many times)
# is_children = 0 
# codes = list ( person_full_icd['ICD9_CODE'] )  
# for k,val in enumerate(codes) : 
  # if (val in parents) and (val not in children) : 
    # is_children = is_children + 1 

# print ('\nfraction of freq. children node over all occurances {}'.format ( is_children*1.0 / len(codes)))

