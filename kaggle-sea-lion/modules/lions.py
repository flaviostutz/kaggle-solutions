import modules.utils as utils
import numpy as np
import cv2
import scipy
from scipy import spatial
import keras
from modules.logging import logger

import tensorflow as tf
from tflearn import layers
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation


#colors in bgr reference
ADULT_MALES = 0
SUBADULT_MALE = 1
ADULT_FEMALE = 2
JUVENILE = 3
PUP = 4

#each index is a min/max color for a class mark
C_MIN = [
            np.array([0, 0, 160]),
            np.array([200, 0, 200]),
            np.array([10, 40, 75]),
            np.array([150, 40, 0]),
            np.array([25, 140, 40])
        ]

C_MAX = [
            np.array([50, 50, 255]),
            np.array([255, 55, 255]),
            np.array([20, 55, 130]),
            np.array([255, 80, 40]),
            np.array([50, 255, 65])
        ]

def find_class(image, point):
    image = image[point[1]-3:point[1]+3,point[0]-3:point[0]+3]
    result = -1
    max = 0
    for col in range(5):
        cmsk = cv2.inRange(image, C_MIN[col], C_MAX[col])
        sm = np.sum(cmsk)
        if(sm!=None and sm>max):
            max = sm
            result = col
    return result


def export_lions(image_raw, image_dotted, target_x_ds, target_y_ds, image_dims, debug=False, min_distance_others=0):
   
    #BLACKOUT PORTIONS OF IMAGE IN RAW PICTURE
    image_dotted_bw = cv2.cvtColor(image_dotted, cv2.COLOR_BGR2GRAY)
    #utils.show_image(image_dotted_bw, size=8)

    mask = cv2.threshold(image_dotted_bw, 5, 255, cv2.THRESH_BINARY)[1]
    #utils.show_image(mask, size=8)

    image_raw_bw = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)
    image_raw = cv2.bitwise_and(image_raw, image_raw, mask=mask)
    #utils.show_image(image_raw, size=8, is_bgr=True)

    
    #ISOLATE HUMAN MARKS ON DOTTED PICTURE
    diff_color = cv2.absdiff(image_dotted, image_raw)
    diff = cv2.cvtColor(diff_color, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((2,2),np.uint8)
    diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, kernel)
    ret,diff = cv2.threshold(diff,10,255,cv2.THRESH_TOZERO)
    ret,diff = cv2.threshold(diff,0,255,cv2.THRESH_BINARY)

    #debug data
    debug_image = image_dotted.copy()
    images = []
    
    #find all dotted sea lions
    count1 = 0
    count_class = np.zeros(5)
    lion_positions = []
    lion_classes = []
    im2, contours, hierarchy = cv2.findContours(diff, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if(w>4 and h>4):
            count1 = count1 + 1
            center = (x+round(w/3),y+round(h/3))
            clazz = find_class(image_dotted, center)
            
            if(clazz==-1):
                logger.warning('could not detect sea lion class at ' + str(center))
                continue

            lion_positions.append(center)
                
            count_class[clazz] = count_class[clazz] + 1
            lion_classes.append(clazz)

            if(debug):
                cv2.circle(debug_image,center,round(w/2),(255,0,0),1)


    #add found sea lions to training dataset
    #filter out lions that are too near each other to minimize noise on training set
    count2 = 0
    count_class_added = np.zeros(5)
    for i, lion_pos in enumerate(lion_positions):

        lion_class = lion_classes[i]
        
        #find distance between current lion and all other lions
        if(min_distance_others>0):
            lp = np.array(lion_pos)
            lp = np.reshape(lp, (1,2))

            dist = spatial.distance.cdist(lp,lion_positions)[0]
            if(len(dist)>1):
                dist = np.sort(dist)[1:]
                if(np.amin(dist)<=min_distance_others):
                    #skip this lion. it is too near others
                    continue

        #export patch to train dataset
        #logger.info('export x, y to dataset. count=' + str(count))
        count2 = count2 + 1
        pw = round(image_dims[1]/2)
        ph = image_dims[1] - pw
        trainX = utils.crop_image_fill(image_raw, (lion_pos[1]-pw,lion_pos[0]-pw), (lion_pos[1]+ph,lion_pos[0]+ph))

        if(debug):
            images.append(trainX)
            cv2.circle(debug_image,center,round(w/2),(0,0,255),2)

        #normalize between 0-1
        #trainX = trainX/255
        target_x_ds.resize((count2, image_dims[0], image_dims[1], image_dims[2]))
        target_x_ds[count2-1:count2] = trainX

        count_class_added[lion_class] = count_class_added[lion_class] + 1
        
        trainY = keras.utils.to_categorical(lion_class, 5)[0]
        target_y_ds.resize((count2, 5))
        target_y_ds[count2-1:count2] = trainY

    if(debug):
        utils.show_image(debug_image, size=8, is_bgr=True)
        utils.show_images(images, cols=10, is_bgr=True, size=1)
        logger.info('total animals found: ' + str(count1))
        logger.info('total animals added to dataset: ' + str(count2))
    
    return count_class, count_class_added


#adapted from alexnet
def convnet_alexnet_lion(image_dims):

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
