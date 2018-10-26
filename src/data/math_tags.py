import os
import constants

DATA_DIR = 'sos-tags-math-sx'
FNAME = 'tags-math-sx-seqs.txt' 
RAW_DATA_FNAME = os.path.join(constants.RAW_PATH, DATA_DIR, FNAME) 


_N_ELEMENTS = 1650

def get_sequence():
    sequences = []
    with open(RAW_DATA_FNAME) as f:
        for line in f:
            sizes, elements = line.split(';')
            
            sequence = []
            sizes = list(map(int, sizes))
            elements = list(map(int, elements))
            idx = 0
            for size in sizes:
                sequence.append(elements[idx:idx+size])
                idx += size
            sequences.append(sequence)
    return sequences

