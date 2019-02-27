import pandas as pd
import numpy as np
from sklearn.utils import resample
import scipy.stats as st
import torch

def to_tensor(obj):
    if type(obj) == np.ndarray:
        return torch.Tensor(obj)
    if type(obj) == pd.core.frame.DataFrame:
        return torch.Tensor(obj.values)
    else:
        return obj

def to_numpy(obj):
    if type(obj) == torch.Tensor:
        return obj.numpy()
    if type(obj) == pd.core.frame.DataFrame:
        return obj.values
    else:
        return obj


def get_onehot_vector(elements, idx_map):
    indices = [idx_map[key] for key in elements if key in idx_map]
    vector = np.zeros(len(idx_map))
    vector[indices] = 1
    return vector

def generate_batch(data, batch_size=500):
    n = len(data)
    for idx in range(0,n, batch_size):
        yield data[idx:idx+batch_size]

def precision_at_k(logits, true_outputs,k):

    count = 0 
    total = 0
    for logit, true_output in zip(logits, true_outputs):
        #ignore first and last because it is not sensible
        logit = logit[:-1]
        true_output = true_output[1:]
        row_idx = np.sort(np.tile(np.arange(0, len(true_output)), k))
        top_k_idx = np.argsort(logit)[:,::-1][:,:k].flatten()
        
        temp = to_numpy(true_output[row_idx, top_k_idx])
        
        count += np.sum(temp)
        total += len(true_output)
        #precisions.append(np.mean(to_numpy(true_output[row_idx, top_k_idx])))
    return count/(total*k)

def bootstrap_CI(stat_fn, k_args, populations, n_bootstrap, confidence=0.95):
    statistics = []
    for _ in range(n_bootstrap):
        samples = resample(*populations)
        statistics.append(stat_fn(*samples, **k_args)) 
    
    return statistics, st.t.interval(confidence, len(statistics)-1, loc=np.mean(statistics), scale=st.sem(statistics))


def generate_data_by_multi_idx(series_df, labels_df):
    indices = set(series_df.index).intersection(set(labels_df.index))
    for idx in indices:
        yield series_df.xs(idx, level=[0,1]), labels_df.xs(idx, level=[0,1])
