import matplotlib.pyplot as plt
import cv2
import numpy as np
import shutil
import os
from time import time
import itertools

from modules.logging import logger

def dataset_xy_balance_classes(input_h5file_path, output_h5file_path):
    #todo

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
        
    ax1.imshow(pixels, cmap=cmap)
    
    if(output_file!=None):
        plt.savefig(output_file)
        plt.close(fig1)
    else:
        plt.show()
    
#def show_slices(pixels, name, nr_slices=12, cols=4, output_dir=None, size=7):
def show_images(image_list, cols=4, name='image', output_dir=None, is_bgr=False, cmap=None, size=6):
    fig = plt.figure()
    rows = round(len(image_list)/cols)+1
    t = Timer('generating image patches. rows=' + str(rows) + '; cols=' + str(cols))
    fig.set_size_inches(cols*size, rows*size)
    for i,im in enumerate(image_list):
        if(is_bgr):
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        y = fig.add_subplot(rows,cols,i+1)
        y.imshow(im, cmap=cmap)

    if(output_dir!=None):
        f = output_dir + name + '.jpg'
        plt.savefig(f)
        plt.close(fig)
    else:
        plt.show()
        
    t.stop()

def dataset_name(name, image_dims):
    return '{}-{}-{}.h5'.format(name, image_dims[0], image_dims[1])

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

def mkdirs(base_dir, dirs=[], recreate=False):
    if(recreate):
        shutil.rmtree(base_dir, True)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    for d in dirs:
        if not os.path.exists(base_dir + d):
            os.makedirs(base_dir + d)
