import h5py

import tensorflow as tf
import tflearn
from tflearn import layers
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
import sklearn

from modules.logging import logger
import modules.logging
import modules.utils as utils
from modules.utils import Timer

_model = None

#adapted from alexnet
def net_alexnet_lion(image_dims):

    #image augmentation
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_flip_updown()
    img_aug.add_random_rotation(max_angle=360.)
    img_aug.add_random_blur(sigma_max=5.)
    
    #image pre-processing
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()
    
    #AlexNet
    network = layers.core.input_data(shape=[None, image_dims[0], image_dims[1], image_dims[2]], dtype=tf.float32, data_preprocessing=img_prep, data_augmentation=img_aug)
    network = layers.conv.conv_2d(network, 96, 11, strides=4, activation='relu')
    network = layers.conv.max_pool_2d(network, 3, strides=2)
    network = layers.normalization.local_response_normalization(network)
    network = layers.conv.conv_2d(network, 256, 5, activation='relu')
    network = layers.conv.max_pool_2d(network, 3, strides=2)
    network = layers.normalization.local_response_normalization(network)
    network = layers.conv.conv_2d(network, 384, 3, activation='relu')
    network = layers.conv.conv_2d(network, 384, 3, activation='relu')
    network = layers.conv.conv_2d(network, 256, 3, activation='relu')
    network = layers.conv.max_pool_2d(network, 3, strides=2)
    network = layers.normalization.local_response_normalization(network)
    network = layers.core.fully_connected(network, 4096, activation='tanh')
    network = layers.core.dropout(network, 0.5)
    network = layers.core.fully_connected(network, 4096, activation='tanh')
    network = layers.core.dropout(network, 0.5)
    network = layers.core.fully_connected(network, 5, activation='softmax')
    network = layers.estimator.regression(network, optimizer='momentum', 
                                          loss='categorical_crossentropy', learning_rate=0.001)
    
    return network


def evaluate_dataset(X, Y, model, batch_size=24, confusion_matrix=False):
    logger.info('Evaluate performance on dataset '+ dataset_path +'...')
    acc = model.evaluate(X, Y, batch_size=batch_size)
    logger.info('Accuracy: ' + str(acc))

    if(confusion_matrix):
        logger.info('Confusion matrix')
        Y_pred = model.predict(X)
        print(sklearn.metrics.confusion_matrix(Y, Y_pred))

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
            
    else:
        logger.info('CNN model already loaded. Reusing it.')
        
    return _model

