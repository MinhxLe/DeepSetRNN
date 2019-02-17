import numpy as np
from sklearn.utils import resample
import scipy.stats as st


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

    precisions = []
    for logit, true_output in zip(logits, true_outputs):
        #ignore first and last because it is not sensible
        logit = logit[:-1]
        true_output = true_output[1:]
        row_idx = np.sort(np.tile(np.arange(0, len(true_output)), k))
        top_k_idx = np.argsort(logit)[:,::-1][:,:k].flatten()
        precisions.append(np.mean(true_output[row_idx, top_k_idx].numpy()))
    return np.mean(precisions)

def bootstrap_CI(stat_fn, k_args, populations, n_bootstrap, confidence=0.95):
    statistics = []
    for _ in range(n_bootstrap):
        samples = resample(*populations)
        statistics.append(stat_fn(*samples, **k_args)) 
    
    return statistics, st.t.interval(confidence, len(statistics)-1, loc=np.mean(statistics), scale=st.sem(statistics))
