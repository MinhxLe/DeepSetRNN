import numpy as np
#basic helper tools


def load(model_dir):
    saver = tf.train.saver()
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir):
        if ckpt and ckpt.model_checkpoint_path:

def iterate_minibatches(inputs,outputs, batch_size, shuffle=False):
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt],outputs[excerpt]

