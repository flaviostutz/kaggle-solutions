import h5py

import tflearn
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.normalization import *
from tflearn.layers.estimator import regression

from modules.logging import logger
import modules.logging
import modules.utils as utils
from modules.utils import Timer
import modules.lungprepare as lungprepare


_model = None

def net_simplest1(image_dims):
    net = input_data(shape=[None, image_dims[0], image_dims[1], image_dims[2], image_dims[3]], dtype=tf.float32)
    
    net = conv_3d(net, 32, 3, strides=1, activation='relu')
    net = max_pool_3d(net, [1,2,2,2,1], strides=[1,2,2,2,1])

    net = conv_3d(net, 64, 3, strides=1, activation='relu')
    net = max_pool_3d(net, [1,2,2,2,1], strides=[1,2,2,2,1])
    
    net = fully_connected(net, 64, activation='relu')
    net = dropout(net, 0.8)
    
    net = fully_connected(net, 2, activation='softmax')
    
    net = regression(net, optimizer='adam',
                     loss='categorical_crossentropy',
                     learning_rate=0.001)
    return net

def evaluate_dataset(dataset_path, model):
    with h5py.File(dataset_path, 'r') as hdf5:
        X = hdf5['X']
        Y = hdf5['Y']
        logger.debug('X_test shape ' + str(X.shape))
        logger.debug('Y_test shape ' + str(Y.shape))
#         for y in Y:
#             print('y=', y)
            
        logger.info('Evaluate performance on dataset '+ dataset_path +'...')
        acc = model.evaluate(X, Y, batch_size=12)
        logger.info('Score: ' + str(acc))

def prepare_model_dirs(output_dir):
    dir_tflogs = output_dir + 'tf-logs'
    dir_checkpoints = output_dir + 'tf-checkpoint'
    dir_checkpoint_best = output_dir + 'tf-checkpoint-best'
    
    logger.info('Preparing output dir')
    utils.mkdirs(output_dir, dirs=['tf-logs'], recreate=False)

    return dir_tflogs, dir_checkpoints, dir_checkpoint_best

def predict_patient(input_dir, patient_id, image_dims, model, output_dir):
    logger.info('>>> Predict patient_id ' + patient_id)
    logger.info('Loading pre-processed images for patient')

    #patient pre-processed image cache
    dataset_file = utils.dataset_path(output_dir, 'cache-predict', image_dims)    
    patient_pixels = None
    with h5py.File(dataset_file, 'a') as h5f:
        try:
            patient_pixels = h5f[patient_id]
            logger.debug('Patient image found in cache. Using it.')
            #disconnect from HDF5
            patient_pixels = np.array(patient_pixels)
            
        except KeyError:
            logger.debug('Patient image not found in cache')
            t = Timer('Preparing patient scan image volume. patient_id=' + patient_id)
            patient_pixels = lungprepare.process_patient_images(input_dir + patient_id, image_dims)
            if(patient_pixels is None):
                logger.warning('Patient lung not found. Skipping.')
            logger.debug('Storing patient image in cache')
            h5f[patient_id] = patient_pixels
            t.stop()
    
    t = Timer('Predicting result on CNN (forward)')
    y = model.predict(np.expand_dims(patient_pixels, axis=0))
    logger.info('PATIENT '+ patient_id +' PREDICT=' + str(y))
    utils.show_slices(patient_pixels, patient_id)
    t.stop()
    
    return y

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

