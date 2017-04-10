import h5py

import tensorflow as tf
import tflearn
from sklearn import metrics
from sklearn import preprocessing
import itertools
import matplotlib.pyplot as plt
import numpy as np

from modules.logging import logger
import modules.logging
import modules.utils as utils
from modules.utils import Timer

_model = None

def evaluate_dataset_tflearn(X, Y, model, batch_size=24, detailed=True, class_labels=None):
    acc = model.evaluate(X, Y, batch_size=batch_size)
    logger.info('Loss: ' + str(acc))

    if(detailed):
        Y_pred = model.predict(X, batch_size=batch_size, verbose=1)

        #we only need the highest probability guess
        Y_pred = np.flip(Y_pred, 1)
        Y_pred = Y_pred[:,0]

        #convert from categorical to label
        lb = preprocessing.LabelBinarizer()
        lb.fit(np.array(range(5)))
        Y = lb.inverse_transform(Y)

        logger.info('Nr test samples: ' + str(len(X)))
        
        logger.info('Kappa score (was this luck?): ' + str(metrics.cohen_kappa_score(Y, Y_pred)))
        
        cm = metrics.confusion_matrix(Y, Y_pred)
        logger.info('Confusion matrix:')
        logger.info(cm)
        
        utils.plot_confusion_matrix(cm, normalize=False)

        
def evaluate_dataset_keras(X, Y, model, batch_size=24, detailed=True, class_labels=None):
    acc = model.evaluate(X, Y, batch_size=batch_size)
    logger.info('Accuracy: ' + str(acc))

    if(detailed):
        Y_pred = model.predict_label(X)

        #we only need the highest probability guess
        Y_pred = np.flip(Y_pred, 1)
        Y_pred = Y_pred[:,0]

        #convert from categorical to label
        lb = preprocessing.LabelBinarizer()
        lb.fit(np.array(range(5)))
        Y = lb.inverse_transform(Y)

        logger.info('Nr test samples: ' + str(len(X)))
        
        logger.info('Kappa score (was this luck?): ' + str(metrics.cohen_kappa_score(Y, Y_pred)))
        
        cm = metrics.confusion_matrix(Y, Y_pred)
        logger.info('Confusion matrix:')
        logger.info(cm)
        
        utils.plot_confusion_matrix(cm, normalize=False)
        
        
def train_batches_keras(X_train, Y_train, X_test, Y_test, epochs=5, datagenerator=None):

    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied)
    datagen.fit(X_train)

    for e in range(epochs):
        print('-'*40)
        print('Epoch', e)
        print('-'*40)
        print("Training...")
        # batch train with realtime data augmentation
        progbar = generic_utils.Progbar(X_train.shape[0])
        for X_batch, Y_batch in buffered_gen_threaded(datagen.flow(X_train, Y_train), buffer_size=buffer_size):
            loss = model.train_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[("train loss", loss)])

        print("Testing...")
        # test time!
        progbar = generic_utils.Progbar(X_test.shape[0])
        for X_batch, Y_batch in buffered_gen_threaded(datagen.flow(X_test, Y_test), buffer_size=buffer_size):
            score = model.test_on_batch(X_batch, Y_batch)
            progbar.add(X_batch.shape[0], values=[("test loss", score)])        
        
def prepare_model_dirs_tflearn(output_dir):
    dir_tflogs = output_dir + 'tf-logs'
    dir_checkpoints = output_dir + 'tf-checkpoint'
    dir_checkpoint_best = output_dir + 'tf-checkpoint-best'
    
    logger.info('Preparing output dir')
    utils.mkdirs(output_dir, dirs=['tf-logs'], recreate=False)

    return dir_tflogs, dir_checkpoints, dir_checkpoint_best

def prepare_cnn_model_tflearn(network, output_dir, model_file=None):
    global _model
    
    if(_model == None):
        
        logger.info('Prepare CNN')
        dir_tflogs, dir_checkpoints, dir_checkpoint_best = prepare_model_dirs(output_dir)

        logger.info('Initializing network...')
        _model = tflearn.models.dnn.DNN(network, tensorboard_verbose=3, 
                                         tensorboard_dir=dir_tflogs,
                                         checkpoint_path=dir_checkpoints,
                                         best_checkpoint_path=dir_checkpoint_best)
        logger.info('Network initialized')

        if(model_file!=None):
            logger.info('Load previous training...')
            _model.load(model_file)
            logger.info('Model loaded')
            
    else:
        logger.info('CNN model already loaded. Reusing it.')
        
    return _model

def show_training_info_keras(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    