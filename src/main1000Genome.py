import os
import numpy as np
import tensorflow as tf
from model import BasicRNNSeq2Seq 
import data
import argparse
import logging
import pandas as pd
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
ap.add_argument('--exp_name',type=str,default="1000Genome_RNN")
#TODO compute this or make it not static
ap.add_argument('--input_size',type=int,default=2)
#ap.add_argument('--timesteps',type=int,default=__) #number SNP 
ap.add_argument('--hidden_size',type=int,default=128)
ap.add_argument('--output_size',type=int,default=3) #TODO know this population is made of less
ap.add_argument('--alpha_contrib',type=float,default=1.0)

#training args
ap.add_argument('--random_seed',type=int,default=42)
ap.add_argument('--load',type=str2bool,default=False)
ap.add_argument('--train',type=str2bool,default=True)
ap.add_argument('--epoch',type=int,default=10)
#ap.add_argument('--batch_size',type=int,default=100)
ap.add_argument('--save_freq',type=int,default=10)
ap.add_argument('--model_name',type=str,default="BasicSeq2Seq")
ap.add_argument('--validate',type=str2bool,default=True)


#hyperparameters
ap.add_argument('--learning_rate',type=float,default=0.01)
ap.add_argument('--momentum',type=float,default=0.9)

#debug/log flags
ap.add_argument('--verbose',type=str2bool,default=True)
ap.add_argument('--debug',type=str2bool,default=True)
ap.add_argument('--log',type=str2bool,default=True)
#directory

#dir flags
ap.add_argument('--models_dir',type=str,default='./model/1000Genome')
ap.add_argument('--logs_dir',type=str,default="./log/1000Genome")
ap.add_argument('--data_dir',type=str,default='./data_test')


def main():
    logger.info(FLAGS)
    model_fname = os.path.join(ckpt_dir,FLAGS.name)
   
    #train test split
    #TODO add cross validation 
    nTrain = int(0.8*len(features))
    X_train = features[:nTrain] 
    y_train = labels[:nTrain] 
    X_test = features[nTrain:] 
    y_test = labels[nTrain:]
    mask_train = relevantMask[:nTrain]
    mask_test = relevantMask[nTrain:]

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
            logger.debug("Begin Training")
            max_epoch = epoch + FLAGS.epoch
            for epoch in range(epoch,max_epoch):
                #validating every save_freq epoch
                if epoch % FLAGS.save_freq == 0: 
                    n_batch = 0
                    accuracy = 0
                    total_loss = 0
                    predict,loss = sess.run([model.prediction,model.loss],
                            feed_dict = { model.X : X_test, model.y : y_test,model.mask : mask_test})
                    predict = np.argmax(predict,axis=2)
                    accuracy = np.mean(np.multiply(predict == y_test,mask_test))
                    logger.info("val:: epoch: {},loss: {},acc: {}".format(epoch,loss,accuracy))
                #optimizing over training data
                _,loss = sess.run([model.optimize, model.loss],
                        feed_dict={ model.X : X_train,model.y : y_train,model.mask : mask_train})
                logger.info("train:: epoch: {}, loss: {}".format(epoch,loss))
                #saving every save_freq epoch
                if epoch % FLAGS.save_freq == 0: 
                    saver.save(sess,model_fname)
                #flushing buffer 
                if FLAGS.log and f_stream:
                    f_stream.flush()
        #final validation
        predict,loss = sess.run([model.prediction,model.loss],
                feed_dict = { model.X : X_test, model.y : y_test,model.mask : mask_test})
        predict = np.argmax(predict,axis=2)
        import pdb;pdb.set_trace()
        accuracy = np.mean(np.multiply(predict == y_test,mask_test))
        mode = "validate_final" if FLAGS.validate else "final"
        logger.info("{}:: loss: {}, accuracy: {}".format(mode,loss,accuracy))
        saver.save(sess,model_fname)
        
        if FLAGS.log and f_stream:
            f_stream.flush()

if __name__ == '__main__':
    #parsing argumentsddddfsdfdadsfdfsdlkajsdlfkj1kjsdflkb
    FLAGS = ap.parse_args()
    
    #experiment and model name
    FLAGS.name = "{}_{}_HS_{}".format(FLAGS.exp_name,\
        FLAGS.model_name,\
        FLAGS.hidden_size)
   
    #setting up logger
    logger = logging.getLogger(FLAGS.name)
    logger.setLevel(logging.INFO)
    if FLAGS.log:
        model_log_fname = os.path.join(FLAGS.logs_dir,FLAGS.name + ".log")
        if not os.path.isdir(FLAGS.logs_dir):
            os.makedirs(FLAGS.logs_dir)
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
    if FLAGS.debug:
        console.setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
    
    #getting dataset
    logger.debug("reading dataset")
    genotypes = pd.read_csv(os.path.join(FLAGS.data_dir,'genotypes.csv'),index_col=0)
    ancestors = pd.read_csv(os.path.join(FLAGS.data_dir,'ancestors.csv'),index_col=0)
    #TODO do we want to drop this column?
    genotypes.drop('CHROM',axis=1,inplace=True)
    ancestors.drop(['CHROM','POS'],axis=1,inplace=True)

    genotypePos = genotypes['POS'].values[:,np.newaxis]
    genotypes = genotypes.values[:,1:] #omit the POS column
    genotypePos = np.broadcast_to(genotypePos,genotypes.shape)
    genotypes = genotypes.T
    genotypePos = genotypePos.T 
    features = np.stack((genotypes,genotypePos),axis=-1)
    labels = ancestors.values.T #individuals by SNP ancestors label
    labels = np.subtract(labels,1) #shifting to ignore 0 and 1
    relevantMask = np.greater(labels,0)
    #cast type
    #TODO save this type in the dataset
    features = features.astype(np.float32) 
    labels = labels.astype(np.int32)
    #debug data
    if FLAGS.debug:
        #work with smaller dataset
        #TODO read in less rows
        features = features[:,:10000,:]
        labels = labels[:,:10000]
        relevantMask = relevantMask[:,:10000]
    #timesteps
    #TODO should not be fixed timestep model(?)
    FLAGS.timesteps = features.shape[1]
    
    #setting up model
    logger.debug("setting up model")
    if FLAGS.model_name == "BasicSeq2Seq":
        model = BasicRNNSeq2Seq(FLAGS)
    elif FLAGS.model_name == "DeepSetSeq2Seq":
        raise ValueError("DeepSetSeq2Seq not implemented yet") 
    else:
        raise ValueError("Expected value model type. inputed %" \
                % FLAGS.model_name)
    
    #checkpoint dir
    ckpt_dir = os.path.join(FLAGS.models_dir, FLAGS.name)
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    #random seed
    np.random.seed(FLAGS.random_seed)
    main()

