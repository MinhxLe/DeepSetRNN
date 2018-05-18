import tensorflow as tf
from tensorflow.contrib import rnn
from ops import linear
import os
from rnn_cell import BasicDeepSetLSTMCell

class BasicRNNClassifier(object):
    def __init__(self,config):
        self.config = config
        #TODO seperate out into functions 
        with tf.variable_scope(config.name):
            config = self.config
            self.X = tf.placeholder(tf.float32,\
                    [None,config.timesteps,config.input_size])
            X = tf.unstack(self.X,config.timesteps,1)
            self.y = tf.placeholder(tf.int32,[None]) 
            
            lstm_cell = rnn.BasicLSTMCell(config.hidden_size,forget_bias=1.0)
            rnn_outputs,states = rnn.static_rnn(lstm_cell,X,dtype=tf.float32)
            output = linear(rnn_outputs[-1],config.output_size)
            #prediction
            self._prediction = tf.nn.softmax(output)
            #loss
            self._loss = tf.reduce_mean((tf.losses.sparse_softmax_cross_entropy(self.y,self._prediction)))
            #optimizer
            #TODO can change optimizer 
            optimizer = tf.train.MomentumOptimizer(\
                    learning_rate=config.learning_rate,\
                    momentum=config.momentum,\
                    use_nesterov=True)
            self._optimize = optimizer.minimize(self._loss)
    @property
    def prediction(self):
        return self._prediction        
    @property
    def loss(self):
        return self._loss
    @property 
    def optimize(self):
        return self._optimize
    @property
    def model_dir(self):
        #TODO add hyperparameters
        #TODO add model name
        return self.config.name

class DeepSetRNNClassifier(object):
    def __init__(self,config):
        self.config = config
        #TODO seperate out into functions 
        with tf.variable_scope(config.name):
            config = self.config
            self.X = tf.placeholder(tf.float32,\
                    [config.batch_size,config.timesteps,config.input_size])
            X = tf.unstack(self.X,config.timesteps,1)
            self.y = tf.placeholder(tf.int32,[config.batch_size]) 
            lstm_cell = BasicDeepSetLSTMCell(config.hidden_size,forget_bias=1.0)
            rnn_outputs,states = rnn.static_rnn(lstm_cell,X,dtype=tf.float32)
            #TODO make this also a deepset(?)
            output = linear(rnn_outputs[-1],config.output_size)
            #prediction
            self._prediction = tf.nn.softmax(output)
            #loss
            self._loss = tf.reduce_mean((tf.losses.sparse_softmax_cross_entropy(self.y,self._prediction)))
            #optimizer
            #TODO can change optimizer 
            optimizer = tf.train.MomentumOptimizer(\
                    learning_rate=config.learning_rate,\
                    momentum=config.momentum,\
                    use_nesterov=True)
            
            self._optimize = optimizer.minimize(self._loss)

    @property
    def prediction(self):
        return self._prediction        

    @property
    def loss(self):
        return self._loss
    
    @property 
    def optimize(self):
        return self._optimize
 
    @property
    def model_dir(self):
        #TODO add hyperparameters
        return self.config.name
