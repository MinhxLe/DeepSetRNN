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
ap.add_argument('--alpha_contrib',type=float,default=1.0)

#training args
ap.add_argument('--random_seed',type=int,default=42)
ap.add_argument('--load',type=str2bool,default=False)
ap.add_argument('--train',type=str2bool,default=True)
ap.add_argument('--epoch',type=int,default=10)
ap.add_argument('--batch_size',type=int,default=100)
ap.add_argument('--save_freq',type=int,default=10)
ap.add_argument('--model_name',type=str)
ap.add_argument('--validate',type=str2bool,default=True)


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


def main():
    logger.info(FLAGS)
    ckpt_dir = os.path.join(FLAGS.models_dir,FLAGS.name)
    model_fname = os.path.join(ckpt_dir,FLAGS.name)
    with tf.Session() as sess:
        saver = tf.train.Saver()
        epoch = 0
        #TODO find a a way to restore previous epoch count

        #loading existing model in if need be
        if FLAGS.load:
            #TODO check if this is successful
            saver.restore(sess,model_fname)
            #epoch = tf.train.global_step(sess)
            logger.info("loading existing model at epoch {}".format(epoch))
        else:
            sess.run(tf.global_variables_initializer())
        if FLAGS.train:
            print(tf.trainable_variables())
            logger.info("Training")
            max_epoch = epoch + FLAGS.epoch
            for epoch in range(epoch,max_epoch):
                #validating every save_freq epoch
                if epoch % FLAGS.save_freq == 0: 
                    n_batch = 0
                    accuracy = 0
                    total_loss = 0
                    for X_batch,y_batch in data.iterate_minibatches(test_images, test_labels,
                            FLAGS.batch_size,True):
                        predict,loss = sess.run([model.prediction,model.loss],
                                feed_dict = { model.X : X_batch, model.y : y_batch})
                        
                        predict = np.argmax(predict,axis=1)
                        accuracy += np.mean(predict == y_batch)
                        total_loss += loss
                        n_batch  += 1
                    logger.info("val:: epoch: {},loss: {},acc: {}".format(epoch,total_loss/n_batch,accuracy/n_batch))
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
                logger.info("train:: epoch: {}, loss: {}".format(epoch,epoch_loss/n_batch))
                #saving every save_freq epoch
                if epoch % FLAGS.save_freq == 0: 
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
        mode = "validate" if FLAGS.validate else "final"
        logger.info("{}:: loss: {}, accuracy: {}".format(mode,loss/n_batch,accuracy/n_batch))
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
    #validate mode
    if FLAGS.validate:
        split = int(len(train_images)*.8)
        test_images = train_images[split:]
        test_labels = train_labels[split:]
        train_images = train_images[:split]
        train_labels = train_labels[:split] 

    #model and model name
    FLAGS.name = "{}_{}_HS_{}".format(FLAGS.exp_name,\
        FLAGS.model_name,\
        FLAGS.hidden_size)
    
    #setting up model
    if FLAGS.model_name == "BasicRNN":
        model = BasicRNNClassifier(FLAGS)
    elif FLAGS.model_name == "DeepSetRNN":
        model = DeepSetRNNClassifier(FLAGS)
        FLAGS.name += "ac_{}".format(FLAGS.alpha_contrib)
    else:
        raise ValueError("Expected value model type. inputed %" \
                % FLAGS.model_name)
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
    
    #random seed
    np.random.seed(FLAGS.random_seed)
    main()

