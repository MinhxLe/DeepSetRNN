import numpy as np

def generate_batch(data, batch_size=500):
    n = len(data)
    for idx in range(0,n, batch_size):
        yield data[idx:idx+batch_size]


def precision_at_k(logits, true_sets,k):
    assert(len(logits) == len(true_sets))
    row_idx = np.sort(np.tile(np.arange(0, len(true_sets)), k))
    top_k_idx = np.argsort(logits)[:,::-1][:,:k].flatten()
    return np.mean(true_sets[row_idx, top_k_idx])


