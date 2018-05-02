import sys
import numpy as np
import os

url_src = 'http://yann.lecun.com/exdb/mnist/'
train_img_fname = "train-images-idx3-ubyte.gz"
train_label_fname = "train-labels-idx1-ubyte.gz"

test_img_fname = "t10k-images-idx3-ubyte.gz"
train_label_name = "t10k-labels-idx1-ubyte.gz"

def __load_data(data_dir,fname,data_type):
    if sys.version_info[0] == 2:
        from urllib import urlretreive
    else:
        from urllib.request import urlretrieve
    def download(fname,local_dir,url_src=url_src):
        print("downloading %s" % fname)
        urlretrieve(url_src + fname, os.path.join(local_dir,fname))
    import gzip
    if not os.path.exists(os.path.join(data_dir,fname)):
        download(fname,data_dir)

    with gzip.open(os.path.join(data_dir,fname),'rb') as f:
        if data_type == 'image':
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        else: #labels
            data = np.frombuffer(f.read(), np.uint8, offset=8)
    return data

def load_MNIST_images(data_dir,test=False):
    if test:
        data = __load_data(data_dir,test_img_fname,'image')
    else:
        data = __load_data(data_dir,train_img_fname,'image')
    data = data.reshape(-1,1,28,28)
    return data

def load_MNIST_labels(data_dir,test=False):
    if test:
        data = __load_data(data_dir,test_img_fname,'labels')
    else:
        data = __load_data(data_dir,train_img_fname,'labels')
    data = data.reshape(-1,1,28,28)
    return data

