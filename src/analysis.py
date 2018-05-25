import os
import numpy as np
import tensorflow as tf
from model import BasicRNNClassifier,DeepSetRNNClassifier
import data
import argparse
import logging

####FLAGS####
ap = argparse.ArgumentParser()

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
#model args
ap.add_argument('--exp_name',type=str,default="MNIST_RNN")
#TODO compute this or make it not static
ap.add_argument('--input_size',type=int,default=28)
ap.add_argument('--timesteps',type=int,default=28) 
ap.add_argument('--hidden_size',type=int,default=128)
ap.add_argument('--output_size',type=int,default=10)

#training args
ap.add_argument('--random_seed',type=int,default=42)
ap.add_argument('--load',type=str2bool,default=True)
ap.add_argument('--train',type=str2bool,default=False)
ap.add_argument('--epoch',type=int,default=10)
ap.add_argument('--batch_size',type=int,default=100)
ap.add_argument('--save_freq',type=int,default=10)
ap.add_argument('--model_name',type=str)

#hyperparameters
ap.add_argument('--learning_rate',type=float,default=0.01)
ap.add_argument('--momentum',type=float,default=0.9)

#debug/log flags
ap.add_argument('--verbose',type=str2bool,default=True)
ap.add_argument('--debug',type=str2bool,default=False)
ap.add_argument('--log',type=str2bool,default=True)
#directory

#dir flags
ap.add_argument('--models_dir',type=str,default='./model')
ap.add_argument('--logs_dir',type=str,default="./log")
ap.add_argument('--data_dir',type=str,default='./data')


#parsing arguments
FLAGS = ap.parse_args()
#getting dataset
train_images = data.load_MNIST_images(FLAGS.data_dir)
test_images  = data.load_MNIST_images(FLAGS.data_dir,True)
train_images = np.squeeze(train_images,axis=1)
test_images = np.squeeze(test_images,axis=1)

train_labels = data.load_MNIST_labels(FLAGS.data_dir)
test_labels  = data.load_MNIST_labels(FLAGS.data_dir,True)

#debug mode
if FLAGS.debug:
    #work with smallete
    console.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    train_images = train_images[0:100]
    test_images = test_images[0:100]
    train_labels = train_labels[0:100]
    test_labels = test_labels[0:100]
    FLAGS.batch_size = 10


FLAGS.name = "{}_{}_HS_{}".format(FLAGS.exp_name,\
    FLAGS.model_name,\
    FLAGS.hidden_size) 
#setting up logger
logger = logging.getLogger(FLAGS.name)
logger.setLevel(logging.INFO)
if FLAGS.log:
    model_log_fname = os.path.join(FLAGS.logs_dir,FLAGS.name + ".log")
    if not os.path.isdir(FLAGS.logs_dir):
        os.makedirs(log_dir)
    #clear log if not loading existing model
    if FLAGS.load:   
        f_stream = logging.FileHandler(model_log_fname,mode='a')
    else:
        f_stream = logging.FileHandler(model_log_fname,mode='w')
    f_stream.setLevel(logging.INFO)
    logger.addHandler(f_stream) 
if FLAGS.verbose:
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)
#checkpoint dir
ckpt_dir = os.path.join(FLAGS.models_dir, FLAGS.name)
if not os.path.isdir(ckpt_dir):
    os.makedirs(ckpt_dir)
#setting up model
if FLAGS.model_name == "BasicRNN":
    model = BasicRNNClassifier(FLAGS)
elif FLAGS.model_name == "DeepSetRNN":
    model = DeepSetRNNClassifier(FLAGS) 
else:
    raise ValueError("Expected value model type. inputed {}".format(FLAGS.model_name))

#random seed
np.random.seed(FLAGS.random_seed)

#analysis
model_fname = os.path.join(ckpt_dir,FLAGS.name)
variables = {}                                                                                     
with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess,model_fname)
    variables_names =[v.name for v in tf.trainable_variables()]
    values = sess.run(variables_names)
    for k,v in zip(variables_names, values):
        variables[k] = v
