import os
import numpy as np
import tensorflow as tf
from model import BasicRNNClassifier,DeepSetRNNClassifier
import data
import argparse
import logging
####FLAGS####
ap = argparse.ArgumentParser()
#model args
ap.add_argument('--name',type=str,default="DeepsetRNN_mean_alpha_zero_init")
ap.add_argument('--input_size',type=int,default=28)
#TODO compute this or make it not static
ap.add_argument('--timesteps',type=int,default=28) 
ap.add_argument('--hidden_size',type=int,default=128)
ap.add_argument('--output_size',type=int,default=10)
#TODO add model type 
#training args
#TODO add initialization parameters
#TODO add random seed
ap.add_argument('--load',type=bool,default=False)
ap.add_argument('--train',type=bool,default=True)
ap.add_argument('--epoch',type=int,default=10)
ap.add_argument('--batch_size',type=int,default=50)
ap.add_argument('--save_freq',type=int,default=10)
#hyperparameters
ap.add_argument('--learning_rate',type=float,default=0.01)
ap.add_argument('--momentum',type=float,default=0.9)
#debug/log flags
ap.add_argument('--verbose',type=bool,default=True)
ap.add_argument('--debug',type=bool,default=False)
ap.add_argument('--log',type=bool,default=True)
#directory
#dir flags
ap.add_argument('--models_dir',type=str,default='./model')
ap.add_argument('--logs_dir',type=str,default="./log")
ap.add_argument('--data_dir',type=str,default='./data')


def main():
    logger.info(FLAGS)
    ckpt_dir = os.path.join(FLAGS.models_dir,FLAGS.name)
    model_fname = os.path.join(ckpt_dir,FLAGS.name)
    with tf.Session() as sess:
        model = DeepSetRNNClassifier(FLAGS)
        saver = tf.train.Saver()
        epoch = 0
        #loading existing model in if need be
        if FLAGS.load:
            #TODO check if this is successful
            saver.restore(sess,model_fname)
            epoch = tf.train.global_step(sess)
            logger.info("loading existing model at epoch {}".format(epoch))
        else:
            sess.run(tf.global_variables_initializer())
        if FLAGS.train:
            print(tf.trainable_variables())
            logger.info("Training")
            max_epoch = epoch + FLAGS.epoch
            for epoch in range(epoch,max_epoch):
                #optimizing over training data
                n_batch = 0
                epoch_loss = 0
                for X_batch, y_batch in data.iterate_minibatches(train_images,train_labels, 
                        FLAGS.batch_size,True):
                    _,loss = sess.run([model.optimize, model.loss],
                            feed_dict={ model.X : X_batch,model.y : y_batch})
                    epoch_loss += loss
                    n_batch += 1
                    logger.debug("epoch:{}, batch:{}, loss:{}".format(epoch,n_batch,loss))
                logger.info("epoch: {}, train_loss: {}".format(epoch,epoch_loss/n_batch))
                #validating and saving every save_freq epoch
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
                    logger.info("epoch: {}, val_acc: {}".format(epoch,accuracy/n_batch))
                    saver.save(sess,model_fname)
                #flushing buffer 
                if FLAGS.log and f_stream:
                    f_stream.flush()
        #final validation
        n_batch = 0
        accuracy = 0
        loss = 0
        for X_batch,y_batch in data.iterate_minibatches(test_images, test_labels,
                FLAGS.batch_size,True):
            predict,batch_loss = sess.run([model.prediction,model.loss],
                    feed_dict = { model.X : X_batch, model.y : y_batch})
            
            predict = np.argmax(predict,axis=1)
            accuracy += np.mean(predict == y_batch)
            loss += batch_loss
            n_batch  += 1
        logger.info("final_loss: {}, final_acc: {}".format(loss/n_batch,accuracy/n_batch))
        saver.save(sess,model_fname)
        
        if FLAGS.log and f_stream:
            f_stream.flush()
if __name__ == '__main__':
    #parsing arguments
    FLAGS = ap.parse_args() 
    #getting dataset
    train_images = data.load_MNIST_images(FLAGS.data_dir)
    test_images  = data.load_MNIST_images(FLAGS.data_dir,True)
    train_images = np.squeeze(train_images,axis=1)
    test_images = np.squeeze(test_images,axis=1)
    
    train_labels = data.load_MNIST_labels(FLAGS.data_dir)
    test_labels  = data.load_MNIST_labels(FLAGS.data_dir,True)
    #TODO setting model name with hyperparameter IF not default 
    #setting up logger
    logger = logging.getLogger(FLAGS.name)
    logger.setLevel(logging.INFO)
    if FLAGS.log:
        model_log_fname = os.path.join(FLAGS.logs_dir,FLAGS.name + ".log")
        if not os.path.isdir(FLAGS.logs_dir):
            os.makedirs(log_dir)
        f_stream = logging.FileHandler(model_log_fname,mode='a')
        f_stream.setLevel(logging.INFO)
        logger.addHandler(f_stream) 
    if FLAGS.verbose:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        logger.addHandler(console)
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
    #checkpoint dir
    ckpt_dir = os.path.join(FLAGS.models_dir, FLAGS.name)
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    
    main()

