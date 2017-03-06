from time import time
from modules.logging import logger
import matplotlib.pyplot as plt
import numpy as np
import h5py
import shutil
import os
import collections

def show_slices(pixels, name, nr_slices=12, cols=4, output_dir=None):
    fig = plt.figure()
    slice_depth = round(np.shape(pixels)[0]/nr_slices)
    rows = round(nr_slices/cols)+1
    fig.set_size_inches(cols*10, rows*10)
    for i in range(nr_slices):
        slice_pos = int(slice_depth*i)
        y = fig.add_subplot(rows,cols,i+1)
        im = pixels[slice_pos]
        if(len(np.shape(im))>2):
            im = im[:,:,0]
        y.imshow(im, cmap='gray')

    if(output_dir!=None):
        f = output_dir + name + '-' + 'slices.jpg'
        plt.savefig(f)
        plt.close(fig)
    else:
        plt.show()

def show_image(pixels, slice_pos, name, output_dir=None):
    fig1, ax1 = plt.subplots(1)
    fig1.set_size_inches(4,4)
    im = pixels[round(np.shape(pixels)[0]*(slice_pos-1))]
    if(len(np.shape(im))>2):
        im = im[:,:,0]
    ax1.imshow(im, cmap=plt.cm.gray)
    
    if(output_dir!=None):
        file = output_dir + name + '-' + 'slice-' + str(slice_pos) + '.jpg'
        plt.savefig(file)
        plt.close(fig1)
    else:
        plt.show()

    
def validate_dataset(dataset_dir, name, image_dims, save_dir=None):
    dataset_file = dataset_path(dataset_dir, name, image_dims)

    ok = True
    logger.info('VALIDATING DATASET ' + dataset_file)

    with h5py.File(dataset_file, 'r') as h5f:
        x_ds = h5f['X']
        y_ds = h5f['Y']

        if(len(x_ds) != len(y_ds)):
            logger.warning('VALIDATION ERROR: x and y datasets with different lengths')
            ok = False

        for px in range(len(x_ds)):
            arr = np.array(x_ds[px])
            if(not np.any(arr)):
                logger.warning('VALIDATION ERROR: Image not found at index=' + str(px))
                ok = False

        label_total = np.array([[0,0]])
        for py in range(len(y_ds)):
            arr = np.array(y_ds[py])
            label_total = arr + label_total
            if(not np.any(arr) or np.all(arr) or arr[0]==arr[1]):
                logger.warning('VALIDATION ERROR: Invalid label found at index=' + str(py) + ' label=' + str(arr))
                ok = False

        label0_ratio = label_total[0][0]/len(y_ds)    
        label1_ratio = label_total[0][1]/len(y_ds)    

        logger.info('Summary')
        logger.info('X shape=' + str(x_ds.shape))
        logger.info('Y shape=' + str(y_ds.shape))
        logger.info('Y: total: ' + str(len(y_ds)))
        logger.info('Y: label 0: ' + str(label_total[0][0]) + ' ' + str(100*label0_ratio) + '%')
        logger.info('Y: label 1: ' + str(label_total[0][1]) + ' ' + str(100*label1_ratio) + '%')
        
        logger.info('Recording sample data')
        size = len(x_ds)
        qtty = min(3, size)
        f = size/qtty
        for i in range(qtty):
            pi = round(i*f)
            logger.info('patient_index ' + str(pi))
            logger.info('x=')
            if(save_dir!=None): 
                mkdirs(save_dir)
                show_slices(x_ds[pi], name + str(y_ds[pi]), output_dir=save_dir)
                logger.info('y=' + str(y_ds[pi]))
    return ok
                
def dataset_path(dataset_dir, name, image_dims):
    return dataset_dir + '{}-{}-{}-{}.h5'.format(name, image_dims[0], image_dims[1], image_dims[2])
                
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
