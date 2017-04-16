import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np
import shutil
import os
from time import time
import h5py
import random
import hashlib
import itertools
from sklearn import preprocessing
from scipy import spatial

from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.io_utils import HDF5Matrix

from sys import stdout

from modules.logging import logger

class BatchGeneratorXYH5:
    """Reads H5 datasets as Python generators. Useful for manipulating datasets that won't fit in memory"""

    def __init__(self, h5file, x_dataset='X', y_dataset='Y', start_ratio=0, end_ratio=1, batch_size=64):
        if(start_ratio>end_ratio):
            raise Exception('End cannot be before start position')
        
        self.h5file = h5file

        self.x_ds = self.h5file[x_dataset]
        self.y_ds = self.h5file[y_dataset]
        self.batch_size = batch_size

        ds_size = len(self.y_ds)

        start_pos = int(np.floor(start_ratio*ds_size))
        end_pos = int(np.ceil(end_ratio*ds_size))
        self.setup_flow(start_pos, end_pos)
        
    def setup_flow(self, start_pos, end_pos):
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.size = end_pos-start_pos
        #self.size = self.size-(self.size%self.batch_size)
        self.nr_batches = int(np.ceil(self.size/self.batch_size))
        
    def flow(self, loop=True):
        counter = 0
        while True:
            if counter == self.nr_batches:
                if(loop and self.nr_batches>0):
                    counter = 0
                else:
                    return

            ds_slice = slice(int(np.floor(self.start_pos + self.batch_size*counter)),int(np.ceil(self.start_pos + self.batch_size*(counter+1))))
            batch_x = self.x_ds[ds_slice]
            batch_y = self.y_ds[ds_slice]
            x_list = []
            y_list = []

            for i,x in enumerate(batch_x):
                y = batch_y[i]
                x_list.append(x)
                y_list.append(y)
            
            x_list = np.array(x_list)
            y_list = np.array(y_list)
            yield x_list, y_list
            counter += 1

    
class ClassBalancerGeneratorXY:
    """Sinks from a xy generator, analyses class distribution and outputs balanced samples. Will undersample and/or augment data if needed to balance classes"""
    
    def __init__(self, source_xy_generator, image_augmentation=None, max_augmentation_ratio=3, max_undersampling_ratio=1, output_weight=1, enforce_max_ratios=False):
        self.source_xy_generator = source_xy_generator

        logger.info('loading input data for class distribution analysis...')

        _, Y_onehot = dump_xy_to_array(source_xy_generator.flow(), source_xy_generator.size, x=False, y=True)
        self.Y_labels = onehot_to_label(Y_onehot)
        self.count_classes = class_distribution(Y_onehot)
        self.nr_classes = np.shape(self.count_classes)[0]
        self.image_augmentation = image_augmentation

        smallest_class = None
        smallest_qtty = 999999999
        largest_class = None
        largest_qtty = 0
        
        logger.info('raw sample class distribution')
        for i,c in enumerate(self.count_classes):
            logger.info(str(i) + ': ' + str(c))
            if(c>0 and c<smallest_qtty):
                smallest_qtty = c
                smallest_class = i
            if(c>largest_qtty):
                largest_qtty = c
                largest_class = i

        minq = largest_qtty - largest_qtty*max_undersampling_ratio
        maxq = smallest_qtty + smallest_qtty*max_augmentation_ratio

        qtty_per_class = max(minq, maxq)
        logger.info('overall output samples per class: ' + str(qtty_per_class))

        logger.info('augmentation/undersampling ratio per class')
        self.ratio_classes = np.zeros(len(self.count_classes))
        for i,c in enumerate(self.count_classes):
            if(c==0):
                self.ratio_classes[i] = 0
            else:
                self.ratio_classes[i] = qtty_per_class/c
            if(enforce_max_ratios):
                if(self.ratio_classes[i]<1):
                    self.ratio_classes[i] = max((1-max_undersampling_ratio), self.ratio_classes[i])
                elif(self.ratio_classes[i]>1):
                    self.ratio_classes[i] = min(1+max_augmentation_ratio, self.ratio_classes[i])

        self.ratio_classes = output_weight * self.ratio_classes
        self.setup_flow(0,1)
    
    def setup_flow(self, output_start_ratio, output_end_ratio, batch_size=64):
        logger.info('SETUP FLOW {} {}'.format(output_start_ratio, output_end_ratio))
        logger.info('output distribution')
        self.output_total_size = 0
        total_samples_ratio = (output_end_ratio-output_start_ratio)
        for i,ratio in enumerate(self.ratio_classes):
            class_total = np.floor(self.count_classes[i]*ratio)
            logger.info(str(i) + ': ' + str(ratio) + ' (' + str(class_total*total_samples_ratio) + ')')
            self.output_total_size += class_total

        logger.info('output total size ' + str(self.output_total_size))
        self.size = int(self.output_total_size * total_samples_ratio)
        
        self.nr_batches = int(np.ceil(self.size/batch_size))
        self.batch_size = batch_size
        logger.info('flow output size ' + str(self.size))
        
        logger.info('calculating source range according to start/end range of the desired output..')
        output_pos = 0
        output_start_pos = int(np.floor(self.output_total_size*output_start_ratio))
        output_end_pos = output_start_pos + self.size
        
        self.source_start_pos = None
        self.source_end_pos = None
        
        for i,y_label in enumerate(self.Y_labels):
            r = self.ratio_classes[y_label]
            if(r==1): 
                output_pos += 1
            elif(r<1):
                if(random.random()<=r):
                    output_pos += 1
            elif(r>1):
                output_pos += r
                
            if(self.source_start_pos==None and output_pos>=output_start_pos):
                self.source_start_pos = i
            
            if(self.source_start_pos!=None and self.source_end_pos==None and output_pos>=output_end_pos):
                self.source_end_pos = i
        
        if(self.source_end_pos==None):
            self.source_end_pos = output_end_pos
        
        logger.info('source range: ' + str(self.source_start_pos) + '-' + str(self.source_end_pos))
        logger.info('output range: ' + str(output_start_pos) + '-' + str(output_end_pos))

        self.source_xy_generator.setup_flow(self.source_start_pos, self.source_end_pos)

    
    def flow(self, max_samples=None, output_dtype='uint8'):
        logger.info('starting new flow...')
        if(np.sum(self.ratio_classes)==0):
            raise StopIteration('no item will be returned by this iterator. aborting')

        x_batch = np.array([], dtype=output_dtype)
        y_batch = np.array([], dtype=output_dtype)

        pending_augmentations = np.zeros(self.nr_classes, dtype='uint32')

        #process each source batch
        count_samples = 0
        for xs,ys in self.source_xy_generator.flow():
            y_labels = onehot_to_label(ys)
            for i,x in enumerate(xs):
                y = ys[i]
            
                if(max_samples!=None and count_samples>=max_samples):
                    break

                label = y_labels[i]
                r = self.ratio_classes[label]

                #add sample
                if(r==1):
                    x_batch,y_batch = self._add_to_batch(x_batch,y_batch,x,y)
#                    logger.info('yielding batch ' + str(len(self.y_batch)) + ' ' + str(self.batch_size))
                    if(len(y_batch)>=self.batch_size):
#                        logger.info('yielding batch1')
                        yield x_batch,y_batch
                        x_batch = np.array([]).astype(output_dtype)
                        y_batch = np.array([]).astype(output_dtype)

                #undersample
                elif(r<1):
                    #accept sample at the rate it should so we balance classes
                    rdm = random.random()
                    if(rdm<=r):
                        x_batch,y_batch = self._add_to_batch(x_batch,y_batch,x,y)
#                        logger.info('yielding batch ' + str(len(self.y_batch)) + ' ' + str(self.batch_size))
                        if(len(y_batch)>=self.batch_size):
#                            logger.info('yielding batch2')
                            yield x_batch,y_batch
                            x_batch = np.array([]).astype(output_dtype)
                            y_batch = np.array([]).astype(output_dtype)

                #augmentation
                elif(r>1):
                    #accept sample
                    x_batch,y_batch = self._add_to_batch(x_batch,y_batch,x,y)
#                    logger.info('yielding batch ' + str(len(self.y_batch)) + ' ' + str(self.batch_size))
                    if(len(y_batch)>=self.batch_size):
#                        logger.info('yielding batch3')
                        yield x_batch,y_batch
                        x_batch = np.array([]).astype(output_dtype)
                        y_batch = np.array([]).astype(output_dtype)
                    
                    pending_augmentations[label] = pending_augmentations[label] + (r-1)
                    pending = int(int(pending_augmentations[label]))

                    #generate augmented copies of images so we balance classes
                    if(pending>0):
                        x1 = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
                        x_orig = np.array([x1])
                        y_orig = np.array([y])

                        #show_image(x_orig[0], is_bgr=False)
                        ir = self.image_augmentation.flow(x_orig, y_orig, batch_size=1)
                        for i in range(pending):
                            it = ir.next()
                            x_it = it[0][0]
                            y_it = it[1]
                            x_it = cv2.cvtColor(x_it, cv2.COLOR_RGB2BGR)
                            
                            x_batch,y_batch = self._add_to_batch(x_batch,y_batch,x,y)
#                            logger.info('yielding batch ' + str(len(self.y_batch)) + ' ' + str(self.batch_size))
                            if(len(y_batch)>=self.batch_size):
                                logger.info('yielding batch4')
                                yield x_batch,y_batch
                                x_batch = np.array([]).astype(self.output_dtype)
                                y_batch = np.array([]).astype(self.output_dtype)
                                
                            pending_augmentations[label] = pending_augmentations[label] - pending

    #x_ds, y_ds: h5py datasets
    def _add_to_batch(self,x_batch,y_batch,x,y):
        if(len(x_batch)==0):
            x_batch = np.resize(x_batch, [0] + list(x.shape))
        x_shape = np.array(x_batch.shape)
        x_shape[0] = x_shape[0] + 1
        x_shape = list(x_shape)
        x_batch = np.resize(x_batch, x_shape)
        x_batch[x_shape[0]-1] = x

        if(len(y_batch)==0):
            y_batch = np.resize(y_batch,[0] + list(y.shape))
        y_shape = np.array(y_batch.shape)
        y_shape[0] = y_shape[0] + 1
        y_shape = list(y_shape)
        y_batch = np.resize(y_batch, y_shape)
        y_batch[y_shape[0]-1] = y
        
        return x_batch, y_batch


                            
def dump_xy_to_dataset(xy_generator, output_h5file, x_dtype='u1', y_dtype='u1', qtty=None):
    """ print('dump train data')
with h5py.File(OUTPUT_DIR + '/test.h5', 'w') as outh5:
    utils.dump_xy_dataset(train_generator, outh5, qtty=12)
    """
    x_ds = None
    y_ds = None
    for xs,ys in xy_generator:
        if(y_ds == None):
            x_ds,y_ds = create_xy_dataset(output_h5file, xs[0].shape, ys[0].shape, x_dtype=x_dtype, y_dtype=y_dtype)
        for i,x in enumerate(xs):
            y = ys[i]
            add_sample_to_dataset(x_ds, y_ds, x, y)
            if(len(y_ds)>=qtty):
                return

def dump_xy_to_array(xy_generator, nr_samples, x=False, y=True, dtype='uint8'):
    """Dump generator contents into a numpy array. Use x and y parameters to avoid dumping too much data from x (or sometimes y)"""
    Xds = np.array([], dtype=dtype)
    Yds = np.array([], dtype=dtype)
    count = 0
    t = Timer('generator dump')
    for x_data,y_data in xy_generator:
        if(count==0):
            s = np.array(np.shape(x_data))
            s[0] = 0
            Xds = np.reshape(Xds, s.tolist())
            s = np.array(np.shape(y_data))
            s[0] = 0
            Yds = np.reshape(Yds, s.tolist())
        count += len(y_data)
        if(x):
            Xds = np.concatenate((Xds, x_data))
            if(len(Xds)>nr_samples):
                Xds = np.split(Xds, [nr_samples])[0]
        if(y):
            Yds = np.concatenate((Yds, y_data))
            if(len(Yds)>=nr_samples):
                Yds = np.split(Yds, [nr_samples])[0]
                
        print_same_line(str(count) + '/' + str(nr_samples))
                
        if(count>=nr_samples):
            break

    t.stop()
    return Xds, Yds
    

def create_xy_dataset(h5file, x_dims, y_dims, x_dtype='u1', y_dtype='u1'):
    x_dims_zero = np.concatenate(([0], np.asarray(x_dims))).tolist()
    x_dims_none = np.concatenate(([None], np.asarray(x_dims))).tolist()

    y_dims_zero = np.concatenate(([0], np.asarray(y_dims))).tolist()
    y_dims_none = np.concatenate(([None], np.asarray(y_dims))).tolist()

    x_ds = h5file.create_dataset('X', x_dims_zero, maxshape=x_dims_none, chunks=True, dtype=x_dtype)
    y_ds = h5file.create_dataset('Y', y_dims_zero, maxshape=y_dims_none, chunks=True, dtype=y_dtype)
    
    return x_ds, y_ds
    
    
#x_ds, y_ds: h5py datasets
def add_sample_to_dataset(x_ds, y_ds, x_data, y_data):
    x_shape = np.array(x_ds.shape)
    x_shape[0] = x_shape[0] + 1
    x_shape = list(x_shape)
    x_ds.resize(x_shape)
    x_ds[x_shape[0]-1] = x_data

    y_shape = np.array(y_ds.shape)
    y_shape[0] = y_shape[0] + 1
    y_shape = list(y_shape)
    y_ds.resize(y_shape)
    y_ds[y_shape[0]-1] = y_data


def image_augmentation_xy(source_batch_generator, image_generator, source_is_bgr=True):
    for items in source_batch_generator:
        xs = []
        ys = []
        for i,bx in enumerate(items[0]):
            by = items[1][i]
            
            if(source_is_bgr):
                bx = cv2.cvtColor(bx, cv2.COLOR_BGR2RGB)
            
            ir = image_generator.flow(np.array([bx]), np.array([by]), batch_size=1)
            im = ir.next()
            bx = im[0][0]
            by = im[1]
            
            if(source_is_bgr):
                bx = cv2.cvtColor(bx, cv2.COLOR_RGB2BGR)
            
            xs.append(bx)
            ys.append(by[0])
        yield np.array(xs), np.array(ys)

        
def print_same_line(log, use_logger=True):
    l = "\r{}".format(log)
    stdout.write(l)
    stdout.flush()
    if(use_logger):
        logger.debug(l)

        
def print_progress(current_value, target_value, elapsed_seconds=None, status=None, size=25, use_logger=True):
    perc = (current_value/target_value)
    pos = round(perc * size)
    s = '{:.0f}/{:.0f} ['.format(current_value, target_value)
    for i in range(pos):
        s = s + '='
    s = s + '>'
    for i in range(pos, size):
        s = s + '.'
    s = s + '] {:d}%'.format(int(perc*100))
    if(elapsed_seconds!=None):
        s = s + ' {:d}s'.format(int(elapsed_seconds))
    if(status!=None):
        s = s + ' ' + str(status)
    print_same_line(s, use_logger=use_logger)

    
def is_distant_from_others(point, other_points, min_distance):
    lp = np.array(point)
    lp = np.reshape(lp, (1,2))

    dist = spatial.distance.cdist(lp,other_points)[0]
    if(len(dist)>1):
        dist = np.sort(dist)[1:]
        if(np.amin(dist)<=min_distance):
            return False
    return True


def onehot_to_label(Y_onehot):
    """Convert from one hot encoding array to class label index
       Y_onehot: numpy array with one hot encoding data (ex.: 0,0,1,0,0). If non array is detected, it will return the input itself
       y = np.array([[1,0,0,0,0,0],[0,1,0,0,0,0],[0,1,0,0,0,0],[0,0,0,0,0,1]])
print(onehot_to_label(y))"""
    if(len(Y_onehot.shape)==1):
        return Y_onehot
    else:
        nr_classes = Y_onehot.shape[1]
        lb = preprocessing.LabelBinarizer()
        lb.fit(np.array(range(nr_classes)))
        return lb.inverse_transform(Y_onehot)

    
#Y_categorical: numpy array with one hot encoding data
def class_distribution(Y_onehot):
    nr_classes = Y_onehot.shape[1]
    count_classes = np.zeros(nr_classes)
    labels = onehot_to_label(Y_onehot)
    for y in labels:
        count_classes[y] = count_classes[y] + 1
    return count_classes.astype('uint32')


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
        plt.text(j, i, '{:.2f}'.format(cm[i, j]),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def dataset_xy_range(h5file, start_ratio, end_ratio):
    X = h5file['X']
    Y = h5file['Y']
    
    s = int(X.shape[0]*start_ratio)
    e = int(X.shape[0]*end_ratio)

    return X[s:e], Y[s:e]

def dataset_xy_hdf5matrix_keras(h5file_path, start_ratio, end_ratio):
    s = None
    e = None
    with h5py.File(h5file_path, 'r') as h5file:
        Y = h5file['Y']
        s = int(Y.shape[0]*start_ratio)
        e = int(Y.shape[0]*end_ratio)
        
    X = HDF5Matrix(h5file_path, 'X', start=s, end=e)
    Y = HDF5Matrix(h5file_path, 'Y', start=s, end=e)
    
    return X, Y

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

def show_images(image_list, image_labels=None, group_by_label=False, cols=4, name='image', output_dir=None, is_bgr=False, cmap=None, size=6):
    logger.info('showing ' + str(len(image_list)) + ' images')
    fig = plt.figure()
    rows = int(len(image_list)/cols)+1
    t = Timer('generating image patches. rows=' + str(rows) + '; cols=' + str(cols))
    fig.set_size_inches(cols*size, rows*size)

    image_indexes = range(len(image_list))

    #order indexes by label
    if(group_by_label==True and image_labels!=None):
        index_label_map = []
        for i,label in enumerate(image_labels):
            index_label_map.append((i,label))
        label_image_map = np.array(index_label_map, dtype=[('index',int),('label',int)])
        label_image_map = np.sort(label_image_map, order='label')
        image_indexes = []
        for a in label_image_map:
            image_indexes.append(a[0])

    c = 0
    for i in image_indexes:
        im = image_list[i]
        if(is_bgr):
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        y = fig.add_subplot(rows,cols,c+1)
        if(cmap==None):
            im = im.astype('uint8')
        y.imshow(im, cmap=cmap)
        
        if(image_labels!=None):
            seed = int(int(hashlib.md5(str(image_labels[i]).encode('utf-8')).hexdigest(),16)/999999999999999999999999999999)
            np.random.seed(seed)
            color = np.random.rand(3,1)
            y.text(4, 17, str(image_labels[i]), fontsize=16, style='normal', bbox={'facecolor':color, 'alpha':1, 'pad':4})
            y.text(4, np.shape(im)[1]-7, '[' + str(i) + ']', fontsize=12, style='normal')
            #y.add_patch(patches.Rectangle((0, 0), np.shape(im)[0]-1, np.shape(im)[1]-1, color=color, linewidth=4, fill=False))
            
        c = c + 1
            
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
        self._lastElapsed = None
    
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
