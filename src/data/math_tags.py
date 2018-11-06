import os
from src.data import constants

DATA_DIR = 'sos-tags-math-sx'
FNAME = 'tags-math-sx-seqs.txt' 
RAW_DATA_FNAME = os.path.join(constants.RAW_PATH, DATA_DIR, FNAME) 


N_ELEMENTS = 1664
def get_sequences():
    sequences = []
    with open(RAW_DATA_FNAME) as f:
        for line in f:
            sizes, elements = line.split(';')
            sizes = sizes.split(',') 
            elements = elements.split(',') 
            
            sequence = []
            sizes = list(map(int, sizes))
            elements = list(map(int, elements))
            idx = 0
            for size in sizes:
                curr_set = elements[idx:idx+size]
                #turing 1idx to 0idx
                curr_set = [i-1 for i in curr_set]
                sequence.append(curr_set)
                idx += size
            sequences.append(sequence)
    return sequences

LABELS_FNAME = os.path.join(constants.RAW_PATH, DATA_DIR, 'tags-math-sx-element-labels.txt')

def get_labels():
    labels = []
    with open(LABELS_FNAME) as f:
        for line in f:
            labels.append(line.split(' ')[1].rstrip())
    return labels
    
