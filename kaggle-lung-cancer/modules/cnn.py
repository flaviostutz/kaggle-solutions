import h5py

import tensorflow as tf
import tflearn
from tflearn import layers
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation
from modules.ImageAugmentation3d import ImageAugmentation3d
from modules.ImagePreprocessing3d import ImagePreprocessing3d
import sklearn

from modules.logging import logger
import modules.logging
import modules.utils as utils
from modules.utils import Timer

_model = None

#as https://github.com/swethasubramanian/LungCancerDetection/blob/master/notebook /LungCancerDetection.ipynb
def net_nodule2d_swethasubramanian(image_dims):

    #image augmentation
    img_aug = ImageAugmentation()
    img_aug.add_random_flip_leftright()
    img_aug.add_random_flip_updown()
    img_aug.add_random_rotation(max_angle=25.)
    img_aug.add_random_blur(sigma_max=3.)
    
    #image pre-processing
    img_prep = ImagePreprocessing()
    img_prep.add_featurewise_zero_center()
    img_prep.add_featurewise_stdnorm()
    
    net = layers.core.input_data(shape=[None, image_dims[0], image_dims[1], image_dims[2], image_dims[3]], dtype=tf.float32, data_preprocessing=img_prep, data_augmentation=img_aug)
    
    net = layers.conv.conv_2d(net, 50, 3, activation='relu')
    net = layers.conv.max_pool_2d(net, 2)
    net = layers.conv.conv_2d(net, 64, 3, activation='relu')
    net = layers.conv.conv_2d(net, 64, 3, activation='relu')
    net = layers.conv.max_pool_2d(net, 2)
    net = layers.core.fully_connected(net, 512, activation='relu')
    net = layers.core.dropout(net, 0.5)
    net = layers.core.fully_connected(net, 2, activation='softmax')

    net = layers.estimator.regression(net, optimizer='adam',
                                      loss='categorical_crossentropy',
                                      learning_rate=0.001)
    return net

#adapted from 2d version
def net_nodule3d_swethasubramanian(image_dims):

    #image augmentation
    img_aug = ImageAugmentation3d()
    img_aug.add_random_flip_x()
    img_aug.add_random_flip_y()
    img_aug.add_random_flip_z()
    img_aug.add_random_rotation(max_angle=25.)
    img_aug.add_random_blur(sigma_max=3.)
    
    #image pre-processing
    img_prep = ImagePreprocessing3d()
    img_prep.add_featurewise_zero_center(mean=-575.756121928)
    img_prep.add_featurewise_stdnorm(std=360.547933391)
    
    net = layers.core.input_data(shape=[None, image_dims[0], image_dims[1], image_dims[2], image_dims[3]], dtype=tf.float32, data_preprocessing=img_prep, data_augmentation=img_aug)
    
    net = layers.conv.conv_3d(net, 50, 3, activation='relu')
    net = layers.conv.max_pool_3d(net, 2)
    net = layers.conv.conv_3d(net, 64, 3, activation='relu')
    net = layers.conv.conv_3d(net, 64, 3, activation='relu')
    net = layers.conv.max_pool_3d(net, 2)
    net = layers.core.fully_connected(net, 512, activation='relu')
    net = layers.core.dropout(net, 0.5)
    net = layers.core.fully_connected(net, 2, activation='softmax')

    net = layers.estimator.regression(net, optimizer='adam',
                                      loss='categorical_crossentropy',
                                      learning_rate=0.001)
    return net



def evaluate_dataset(dataset_path, model, batch_size=12, confusion_matrix=False):
    with h5py.File(dataset_path, 'r') as hdf5:
        X = hdf5['X']
        Y = hdf5['Y']
        logger.debug('X_test shape ' + str(X.shape))
        logger.debug('Y_test shape ' + str(Y.shape))
        
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

        if(model_file!=None):
            logger.info('Load previous training...')
            _model.load(model_file)
            
    else:
        logger.info('CNN model already loaded. Reusing it.')
        
    return _model

