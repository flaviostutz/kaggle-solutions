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

import keras

_model = None

class LoggingLogger(keras.callbacks.Callback):

    def __init__(self):
        super(LoggingLogger,self).__init__()
        self.epoch = 0

    def on_train_begin(self, logs=None):
        self.verbose = self.params['verbose']
        self.epochs = self.params['epochs']

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.target = self.params['steps']

    def on_batch_begin(self, batch, logs=None):
        logger.debug('batch ' + str(batch) + '/' + str(self.target))
        logger.debug('epoch ' + str(self.epoch) + '/' + str(self.epochs))
        for k in self.params['metrics']:
            if k in logs:
                logger.debug(str(k) + '=' + str(logs[k]))

    def on_batch_end(self, batch, logs=None):
        pass
                
    def on_epoch_end(self, epoch, logs=None):
        pass

def show_predictions(xy_generator, qtty, model, is_bgr=True, group_by_label=False, size=1.4):
    x, y = utils.dump_xy_to_array(xy_generator, qtty, x=True, y=True)
    y_pred = model.predict(x)

    yl = utils.onehot_to_label(np.array(y))
    ylp = utils.onehot_to_label(np.array(y_pred))

    labels = [(lambda a,b: str(a) + '/' + str(b))(y,yp) for y,yp in zip(yl,ylp)]

    utils.show_images(x, image_labels=labels, cols=12, is_bgr=is_bgr, group_by_label=group_by_label, size=size)
    
    
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

        
def evaluate_dataset_keras(xy_generator, nr_batches, nr_samples, model, detailed=True, class_labels=None):
    logger.info('Evaluating model performance (' + str(nr_samples) + ' samples)...')
    acc = model.evaluate_generator(xy_generator, nr_batches)
    logger.info('Accuracy: ' + str(acc[1]) + ' - Loss: ' + str(acc[0]))

    if(detailed):
        logger.info('Predicting Y for detailed analysis...')
        Y_pred = model.predict_generator(xy_generator, nr_batches+1)
        #sometimes predict_generator returns more samples than nr_batches*batch_size
        Y_p = np.array(np.split(Y_pred, [nr_samples]))[0]

        #we only need the highest probability guess
        Y_pred = np.swapaxes(Y_pred, 0, 1)
        Y_pred = np.argmax(Y_p, axis=1)

        #convert from categorical to label
        _, Y = utils.dump_xy_to_array(xy_generator, nr_samples)
        if(len(Y)>0):
            lb = preprocessing.LabelBinarizer()
            lb.fit(np.array(range(np.shape(Y[0])[0])))
            Y = lb.inverse_transform(Y)

            logger.info('Nr test samples: ' + str(len(Y)) + '|' + str(len(Y_pred)))

            logger.info('Kappa score (was this luck?): ' + str(metrics.cohen_kappa_score(Y, Y_pred)))

            cm = metrics.confusion_matrix(Y, Y_pred)
            logger.info('Confusion matrix:')
            logger.info(cm)
        
            utils.plot_confusion_matrix(cm, normalize=False)
        else:
            logger.info('No samples found in xy_generator')


def get_callbacks_keras(model, weights_dir, tf_logs_dir):
    weights_file = weights_dir + 'weights-{epoch:02d}-{val_acc:.2f}.h5'
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=tf_logs_dir, histogram_freq=0, write_graph=True, write_images=True)
    tensorboard_callback.set_model(model)
    checkpoint_callback = keras.callbacks.ModelCheckpoint(weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    progbar_callback = keras.callbacks.ProgbarLogger(count_mode='steps')
    logger_callback = LoggingLogger()
    
        
        
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
    fig = plt.figure()
    fig.set_size_inches(8, 3)

    plt.subplot(121)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    #plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    
    # summarize history for loss
    plt.subplot(122)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    #plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validate'], loc='upper left')
    
    plt.show()
    