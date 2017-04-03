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

def evaluate_dataset(X, Y, model, batch_size=24, detailed=True, class_labels=None):
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
    
def prepare_model_dirs(output_dir):
    dir_tflogs = output_dir + 'tf-logs'
    dir_checkpoints = output_dir + 'tf-checkpoint'
    dir_checkpoint_best = output_dir + 'tf-checkpoint-best'
    
    logger.info('Preparing output dir')
    utils.mkdirs(output_dir, dirs=['tf-logs'], recreate=False)

    return dir_tflogs, dir_checkpoints, dir_checkpoint_best

def prepare_cnn_model(network, output_dir, model_file=None):
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

