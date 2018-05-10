import os
import numpy as np
import tensorflow as tf
from model import BasicRNNClassifier
import data
import logging
#TODO add logging
#TODO add argparse
#TODO fix flags
####FLAGS####
flags = tf.app.flags
#model flags
#TODO edit help message
flags.DEFINE_string('name',"BasicRNN", "Model Name")
flags.DEFINE_integer("input_size", 28, "Epoch to train [30]")
flags.DEFINE_integer("timesteps", 28, "Epoch to train [30]")
flags.DEFINE_integer("hidden_size", 128, "Epoch to train [30]")
flags.DEFINE_integer("output_size",10, "Epoch to train [30]")
#training flags
flags.DEFINE_boolean("load", False, "True for loading prior model, \
        False for testing [False]")
flags.DEFINE_boolean("train", True, "True for training, \
        False for testing [False]")
flags.DEFINE_integer("epoch", 100, "Epoch to train [30]")
flags.DEFINE_float("learning_rate", 0.01, "Learning rate")
flags.DEFINE_float("momentum", 0.9, "momentum")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [200]")
#dir flags
flags.DEFINE_string("models_dir", "../model", \
        "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("data_dir", "../data", 
        "Directory name to save the checkpoints [checkpoint]")
FLAGS = flags.FLAGS

#debug/log flags
flags.DEFINE_boolean("debug", True, "debug mode")
flags.DEFINE_integer("save_freq",10, "Epoch to train [30]")

####LOGGING####
logger = logging.getLogger(FLAGS.name)
logger.setLevel(logging.INFO)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logger.addHandler(console)

def main(_):
    logger.info(FLAGS)
    #reading data
    model_dir = os.path.join(FLAGS.models_dir,FLAGS.name)
    with tf.Session() as sess:
        model = BasicRNNClassifier(FLAGS)
        saver = tf.train.Saver()
        #load
        epoch = 0
        if FLAGS.load:
            #TODO move to seperate function
            logger.info("loading existing model")
            saver.restore(sess,os.path.join(model_dir,FLAGS.name))
            epoch = tf.train.global_step(sess)
        else:
            sess.run(tf.global_variables_initializer())
        #train 
        if not os.path.isdir(model_dir):
            os.makedirs(model_dir)
        if FLAGS.train:
            logger.info("training")
            max_epoch = epoch + FLAGS.epoch
            for epoch in range(epoch,max_epoch):
                #validating and saving every frequence
                if epoch % FLAGS.save_freq == 0: 
                    n_batch = 0
                    accuracy = 0
                    for X_batch,y_batch in data.iterate_minibatches(test_images, test_labels,
                            FLAGS.batch_size,True):
                        predict,loss = sess.run([model.prediction,model.loss],
                                feed_dict = { model.X : X_batch, model.y : y_batch})
                        
                        predict = np.argmax(predict,axis=1)
                        accuracy += np.mean(predict == y_batch)
                        n_batch  += 1
                    logger.info("validation acc: {}".format(accuracy/n_batch))
                    saver.save(sess,os.path.join(model_dir,FLAGS.name))
                
                n_batch = 0
                epoch_loss = 0
                for X_batch, y_batch in data.iterate_minibatches(train_images,train_labels, 
                        FLAGS.batch_size,True):
                    _,loss = sess.run([model.optimize, model.loss],
                            feed_dict={ model.X : X_batch,model.y : y_batch})
                    epoch_loss += loss
                    n_batch += 1
                logger.info("epoch:{} loss:{}".format(epoch,epoch_loss/n_batch))
if __name__ == '__main__':
    #getting dataset
    train_images = data.load_MNIST_images(FLAGS.data_dir)
    test_images  = data.load_MNIST_images(FLAGS.data_dir,True)
    train_images = np.squeeze(train_images,axis=1)
    test_images = np.squeeze(test_images,axis=1)
    
    train_labels = data.load_MNIST_labels(FLAGS.data_dir)
    test_labels  = data.load_MNIST_labels(FLAGS.data_dir,True)
    
    tf.app.run()
