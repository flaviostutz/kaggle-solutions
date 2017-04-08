import matplotlib.pyplot as plt
import cv2
import numpy as np
import shutil
import os
from time import time
import h5py
import random
import itertools
from sklearn import preprocessing
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from modules.logging import logger

#x_ds, y_ds: h5py datasets
def add_sample_to_dataset(x_ds, y_ds, x_data, y_data):
    x_shape = np.array(x_ds.shape)
    x_shape[0] = x_shape[0] + 1
    x_shape = list(x_shape)
    x_ds.resize(x_shape)
    x_ds[x_shape[0]-1] = x_data

    #show_image(x_data, is_bgr=True)
    #show_image((x_ds[x_shape[0]-1]).astype('uint8'), is_bgr=True)

    y_shape = np.array(y_ds.shape)
    y_shape[0] = y_shape[0] + 1
    y_shape = list(y_shape)
    y_ds.resize(y_shape)
    y_ds[y_shape[0]-1] = y_data

#convert from categorical to label
#Y_categorical: numpy array with one hot encoding data (ex.: 0,0,1,0,0)
def categorical_to_label(Y_categorical):
    nr_classes = Y_categorical.shape[1]
    lb = preprocessing.LabelBinarizer()
    lb.fit(np.array(range(nr_classes)))
    return lb.inverse_transform(Y_categorical)

#Y_categorical: numpy array with one hot encoding data
def class_distribution(Y_categorical):
    nr_classes = Y_categorical.shape[1]
    count_classes = np.zeros(nr_classes)
    labels = categorical_to_label(Y_categorical)
    for y in labels:
        count_classes[y] = count_classes[y] + 2
    return count_classes

def create_xy_dataset(h5file, x_dims, y_dims):
    x_dims_zero = np.concatenate(([0], np.asarray(x_dims))).tolist()
    x_dims_none = np.concatenate(([None], np.asarray(x_dims))).tolist()

    y_dims_zero = np.concatenate(([0], np.asarray(y_dims))).tolist()
    y_dims_none = np.concatenate(([None], np.asarray(y_dims))).tolist()

    x_ds = h5file.create_dataset('X', x_dims_zero, maxshape=x_dims_none, chunks=True, dtype='f')
    y_ds = h5file.create_dataset('Y', y_dims_zero, maxshape=y_dims_none, chunks=True, dtype='f')
    
    return x_ds, y_ds

#max_augmentation_rotation=20, max_augmentation_shift=0, max_augmentation_scale=1, augmentation_flip_leftright=True, augmentation_flip_updown=True
def dataset_xy_balance_classes_image(input_h5file, output_h5file, max_augmentation_ratio=3, max_undersampling_ratio=1, classes_distribution_weight=1, enforce_max_ratios=False, image_data_generator=None):
    if(image_data_generator==None):
        image_data_generator = ImageDataGenerator(
            rotation_range=360,
            fill_mode='wrap',
            cval=127,
            data_format=K.image_data_format())

    input_x_ds = input_h5file['X']
    input_y_ds = input_h5file['Y']
    if(len(input_x_ds)==0):
        raise Exception('No data found on input dataset')
    x_dims = input_x_ds.shape[1:]
    y_dims = input_y_ds.shape[1:]

    nr_classes = input_y_ds.shape[1]

    t = Timer('traversing entire dataset in order to extract population classes distribution')
    count_classes = class_distribution(input_y_ds[()])
    t.stop()

    logger.info('population distribution')
    smallest_class = None
    smallest_qtty = 999999999
    largest_class = None
    largest_qtty = 0
    for i,c in enumerate(count_classes):
        logger.info(str(i) + ': ' + str(c))
        if(c<smallest_qtty):
            smallest_qtty = c
            smallest_class = i
        if(c>largest_qtty):
            largest_qtty = c
            largest_class = i
    
    minq = largest_qtty - largest_qtty*max_undersampling_ratio
    maxq = smallest_qtty + smallest_qtty*max_augmentation_ratio

    qtty_per_class = max(minq, maxq)
    logger.info('targeting items per class: ' + str(qtty_per_class))

    logger.info('augmentation/undersampling ratio per class')
    ratio_classes = np.zeros(nr_classes)
    for i,c in enumerate(count_classes):
        if(c==0):
            raise Exception('Class ' + str(i) + ' has zero samples. Aborting class balancing')
        ratio_classes[i] = qtty_per_class/c
        if(enforce_max_ratios):
            if(ratio_classes[i]<1):
                ratio_classes[i] = max((1-max_undersampling_ratio), ratio_classes[i])
            elif(ratio_classes[i]>1):
                ratio_classes[i] = min(1+max_augmentation_ratio, ratio_classes[i])
        logger.info(str(i) + ': ' + str(ratio_classes[i]))

    ratio_classes = classes_distribution_weight * ratio_classes
    logger.info(str(i) + ': ' + str(ratio_classes[i]))

    output_x_ds, output_y_ds = create_xy_dataset(output_h5file, x_dims, y_dims)

    logger.info('iterating over input dataset for generating a new balanced dataset using undersampling and/or augmentation')
    y_labels = categorical_to_label(input_y_ds[()])

    pending_augmentations = np.zeros(nr_classes)
            
    for i,x in enumerate(input_x_ds):
        y = input_y_ds[i]
        #x = input_x_ds[i]
        label = y_labels[i]
        r = ratio_classes[label]

        #add sample
        if(r==1):
            add_sample_to_dataset(output_x_ds, output_y_ds, x, y)
               
        #undersample
        if(r<1):
            #accept sample at the rate it should so we balance classes
            rdm = random.random()
            if(rdm<=r):
                add_sample_to_dataset(output_x_ds, output_y_ds, x, y)

        #augmentation
        elif(r>1):
            #accept sample
            add_sample_to_dataset(output_x_ds, output_y_ds, x, y)
            pending_augmentations[label] = pending_augmentations[label] + (r-1)

            pending = int(round(pending_augmentations[label]))

            #generate augmented copies of images so we balance classes
            if(pending>0):
                x1 = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                x_orig = np.array([x1])
                y_orig = np.array([y])

                #show_image(x_orig[0], is_bgr=False)
                ir = image_data_generator.flow(x_orig, y_orig, batch_size=1)
                for i in range(pending):
                    it = ir.next()
                    x_it = it[0][0]
                    y_it = it[1]
                    x_it = cv2.cvtColor(x_it, cv2.COLOR_RGB2BGR)
                    add_sample_to_dataset(output_x_ds, output_y_ds, x_it, y_it)
                    #show_image(x_it, is_bgr=True)

            pending_augmentations[label] = pending_augmentations[label] - pending
    
    logger.info('done')
    

def plot_confusion_matrix(cm, class_labels=None,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    if(class_labels==None):
        class_labels = ["{:d}".format(x) for x in range(len(cm))]
    tick_marks = np.arange(len(class_labels))
    plt.xticks(tick_marks, class_labels, rotation=45)
    plt.yticks(tick_marks, class_labels)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def dataset_xy_range(h5file, start_ratio, end_ratio):
    X = h5file['X']
    Y = h5file['Y']
    
    s = round(X.shape[0]*start_ratio)
    e = round(X.shape[0]*end_ratio)

    return X[s:e], Y[s:e]

#this function crops image and handles edge conditions
def crop_image_fill(image, p1, p2, fill_color=127):
    s = np.shape(image)
    nim = np.full((p2[0]-p1[0], p2[1]-p1[1], 3), 127, dtype=np.uint8)
    cim = image[p1[0]:p2[0],p1[1]:p2[1]]
    sc = np.shape(cim)
    nim[0:sc[0],0:sc[1]] = cim
    return nim

def show_image(pixels, output_file=None, size=6, is_bgr=False, cmap=None):
    fig1, ax1 = plt.subplots(1)
    fig1.set_size_inches(size,size)
    if(is_bgr):
        pixels = cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)

    if(cmap==None):
        pixels = pixels.astype('uint8')
        
    ax1.imshow(pixels, cmap=cmap)
    
    if(output_file!=None):
        plt.savefig(output_file)
        plt.close(fig1)
    else:
        plt.show()

#def show_slices(pixels, name, nr_slices=12, cols=4, output_dir=None, size=7):
def show_images(image_list, image_labels=None, cols=4, name='image', output_dir=None, is_bgr=False, cmap=None, size=6):
    logger.info('showing ' + str(len(image_list)) + ' images')
    fig = plt.figure()
    rows = round(len(image_list)/cols)+1
    t = Timer('generating image patches. rows=' + str(rows) + '; cols=' + str(cols))
    fig.set_size_inches(cols*size, rows*size)
    for i,im in enumerate(image_list):
        if(is_bgr):
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        y = fig.add_subplot(rows,cols,i+1)
        if(cmap==None):
            im = im.astype('uint8')
        y.imshow(im, cmap=cmap)
        
        if(image_labels!=None):
            np.random.seed(image_labels[i]+1)
            circ = plt.Circle((5, 5), 10, color=np.random.rand(3,1))
            y.add_patch(circ)
    
    if(output_dir!=None):
        f = output_dir + name + '.jpg'
        plt.savefig(f)
        plt.close(fig)
    else:
        plt.show()
        
    t.stop()

def dataset_name(name, image_dims):
    return '{}-{}-{}.h5'.format(name, image_dims[0], image_dims[1])

def mkdirs(base_dir, dirs=[], recreate=False):
    if(recreate):
        shutil.rmtree(base_dir, True)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    for d in dirs:
        if not os.path.exists(base_dir + d):
            os.makedirs(base_dir + d)

class Timer:
    def __init__(self, name, debug=True):
        self._name = name
        self._debug = debug
        self.start()
    
    def start(self):
        self._start = time()
        if(self._debug):
            logger.info('> [started] ' + self._name + '...')

    def stop(self):
        self._lastElapsed = (time()-self._start)
        if(self._debug):
            logger.info('> [done]    {} ({:.3f} ms)'.format(self._name, self._lastElapsed*1000))
            
    def elapsed(self):
        if(self._lastElapsed != None):
            return (self._lastElapsed)
        else:
            return (time()-self._start)
