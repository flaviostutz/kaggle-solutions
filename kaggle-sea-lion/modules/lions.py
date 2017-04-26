import modules.utils as utils
import numpy as np
import cv2
import scipy
import keras
from modules.logging import logger
import modules.utils as utils
import random

import tensorflow as tf
import keras
from keras import models
from keras import layers
from keras.layers import convolutional
from keras.layers import core

CLASS_LABELS = ['0-adult_male', '1-subadult_male', '2-adult_female', '3-juvenile', '4-pup', '5-non lion']

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


#adapted from alexnet
def convnet_alexnet_lion_keras(image_dims):
#    model = Sequential()
#    model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=image_dims))

    NR_CLASSES = 6

    input = layers.Input(shape=image_dims, name="Input")
    conv_1 = convolutional.Convolution2D(96, 11, 11, border_mode='valid', name="conv_1", activation='relu', init='glorot_uniform')(input)
    pool_1 = convolutional.MaxPooling2D(pool_size=(3, 3), name="pool_1")(conv_1)
    zero_padding_1 = convolutional.ZeroPadding2D(padding=(1, 1), name="zero_padding_1")(pool_1)
    conv_2 = convolutional.Convolution2D(256, 3, 3, border_mode='valid', name="conv_2", activation='relu', init='glorot_uniform')(zero_padding_1)
    pool_2 = convolutional.MaxPooling2D(pool_size=(3, 3), name="pool_2")(conv_2)
    zero_padding_2 = keras.layers.convolutional.ZeroPadding2D(padding=(1, 1), name="zero_padding_2")(pool_2)
    conv_3 = convolutional.Convolution2D(384, 3, 3, border_mode='valid', name="conv_3", activation='relu', init='glorot_uniform')(zero_padding_2)
    conv_4 = convolutional.Convolution2D(384, 3, 3, border_mode='valid', name="conv_4", activation='relu', init='glorot_uniform')(conv_3)
    conv_5 = convolutional.Convolution2D(256, 3, 3, border_mode='valid', name="conv_5", activation='relu', init='glorot_uniform')(conv_4)
    pool_3 = convolutional.MaxPooling2D(pool_size=(3, 3), name="pool_3")(conv_5)
    flatten = core.Flatten(name="flatten")(pool_3)
    fc_1 = core.Dense(4096, name="fc_1", activation='relu', init='glorot_uniform')(flatten)
    fc_1 = core.Dropout(0.5, name="fc_1_dropout")(fc_1)
    output = core.Dense(4096, name="Output", activation='relu', init='glorot_uniform')(fc_1)
    output = core.Dropout(0.5, name="Output_dropout")(output)
    fc_2 = core.Dense(NR_CLASSES, name="fc_2", activation='softmax', init='glorot_uniform')(output)

    return models.Model([input], [fc_2])

def convnet_medium_lion_keras(image_dims):
    model = keras.models.Sequential()

    model.add(core.Lambda(lambda x: (x / 255.0) - 0.5, input_shape=image_dims))

    model.add(convolutional.Conv2D(128, (3, 3), activation='relu', padding='same', init='glorot_uniform'))
    model.add(convolutional.MaxPooling2D(pool_size=(2,2)))
    model.add(convolutional.Conv2D(256, (3, 3), activation='relu', padding='same', init='glorot_uniform'))
    model.add(convolutional.Conv2D(256, (3, 3), activation='relu', padding='same', init='glorot_uniform'))
    model.add(convolutional.MaxPooling2D(pool_size=(2,2)))

    model.add(core.Flatten())

    model.add(core.Dense(1024, activation='relu', init='glorot_uniform'))
    model.add(core.Dropout(0.5))
    model.add(core.Dense(2048, activation='relu', init='glorot_uniform'))
    model.add(core.Dropout(0.5))
    model.add(core.Dense(6, activation='softmax', init='glorot_uniform'))
    
    return model


#don't change. there are already good train for this net (72% acc)
def convnet_simple_lion_keras(image_dims):
    model = keras.models.Sequential()

    model.add(core.Lambda(lambda x: (x / 255.0) - 0.5, input_shape=image_dims))

    model.add(convolutional.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(convolutional.MaxPooling2D(pool_size=(2,2)))
    model.add(convolutional.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(convolutional.MaxPooling2D(pool_size=(2,2)))
    model.add(convolutional.Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(convolutional.MaxPooling2D(pool_size=(2,2)))

    model.add(core.Flatten())

    model.add(core.Dense(512, activation='relu'))
    model.add(core.Dropout(0.5))
    model.add(core.Dense(1024, activation='relu'))
    model.add(core.Dropout(0.5))
    model.add(core.Dense(6, activation='softmax'))
    
    return model


#adapted from alexnet
def convnet_alexnet_lion_tflearn(image_dims):

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


def export_lions(image_raw, image_dotted, target_x_ds, target_y_ds, image_dims, debug=False, min_distance_others=50, non_lion_distance=150):
    
    NR_CLASSES = 6
   
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
    count_class = np.zeros(NR_CLASSES)
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


    count_class_added = np.zeros(NR_CLASSES)

    #add found sea lions to training dataset
    #filter out lions that are too near each other to minimize noise on training set
    count2 = 0
    for i, lion_pos in enumerate(lion_positions):

        lion_class = lion_classes[i]

        is_distant = True
        if(min_distance_others>0):
            is_distant = utils.is_distant_from_others(lion_pos, lion_positions, min_distance_others)

        if(is_distant):
            #export patch to train dataset
            count2 = count2 + 1
            pw = round(image_dims[1]/2)
            ph = image_dims[1] - pw
            #trainX = image_raw[lion_pos[1]-pw:lion_pos[1]+ph,lion_pos[0]-pw:lion_pos[0]+ph]
            trainX = utils.crop_image_fill(image_raw, (lion_pos[1]-pw,lion_pos[0]-pw), (lion_pos[1]+ph,lion_pos[0]+ph))

            m = np.mean(trainX)
            
            if(m>30 and m<225 and m!=127):

                if(debug):
                    images.append(trainX)
                    cv2.circle(debug_image,lion_pos,round(w/2),(0,0,255),2)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(debug_image,str(lion_class),lion_pos, font, 1.1,(255,255,255),2,cv2.LINE_AA)

                #normalize between 0-1
                #trainX = trainX/255
                trainY = keras.utils.to_categorical([lion_class], NR_CLASSES)[0]
                utils.add_sample_to_dataset(target_x_ds, target_y_ds, trainX, trainY)
                count_class_added[lion_class] = count_class_added[lion_class] + 1
        
    #identify non sea lion patches
    count3 = 0
    s = np.shape(image_raw)
    for i in range(int(count2*1.1)):
        patch_pos = (random.randint(image_dims[1]*2, s[1]-image_dims[1]*2), random.randint(image_dims[0]*2, s[0]-image_dims[0]*2))
        is_distant = utils.is_distant_from_others(patch_pos, lion_positions, non_lion_distance)

        if(is_distant):
            #export patch to train dataset
            pw = round(image_dims[1]/2)
            ph = image_dims[1] - pw
            #trainX = image_raw[lion_pos[1]-pw:lion_pos[1]+ph,lion_pos[0]-pw:lion_pos[0]+ph]
            trainX = utils.crop_image_fill(image_raw, (patch_pos[1]-pw,patch_pos[0]-pw), (patch_pos[1]+ph,patch_pos[0]+ph))

            m = np.mean(trainX)
            
            if(m>50 and m<200):
                count3 = count3 + 1
                if(debug):
                    images.append(trainX)
                    cv2.circle(debug_image,patch_pos,round(w/2),(0,255,0),3)

                #normalize between 0-1
                #trainX = trainX/255
                trainY = keras.utils.to_categorical([5], NR_CLASSES)[0]
                utils.add_sample_to_dataset(target_x_ds, target_y_ds, trainX, trainY)
                count_class[5] = count_class[5] + 1
                count_class_added[5] = count_class_added[5] + 1

    logger.info('sea lions found: ' + str(count1))
    logger.info('sea lions added to dataset: ' + str(count2))
    logger.info('non sea lions added to dataset: ' + str(count3))
    logger.info('dataset size: ' + str(len(target_x_ds)))
                
    if(debug):
        utils.show_image(debug_image, size=40, is_bgr=True)
        utils.show_images(images, cols=10, is_bgr=True, size=1.5)
    
    return count_class, count_class_added
