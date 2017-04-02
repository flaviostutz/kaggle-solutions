import modules.utils as utils
import numpy as np
import cv2
from tflearn.data_utils import to_categorical
from modules.logging import logger

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
    result = 'none'
    max = 0
    for col in range(5):
        cmsk = cv2.inRange(image, C_MIN[col], C_MAX[col])
        sm = np.sum(cmsk)
        if(sm>max):
            max = sm
            result = col
    return result


def export_lions(image_raw, image_dotted, target_x_ds, target_y_ds, image_dims, debug=False):
   
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
    
    count = 0
    count_class = np.zeros(5)
    im2, contours, hierarchy = cv2.findContours(diff, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x,y,w,h = cv2.boundingRect(c)
        if(w>4 and h>4):
            count = count + 1
            center = (x+round(w/3),y+round(h/3))
            clazz = find_class(image_dotted, center)
            #logger.info('found=' + str(clazz))
            count_class[clazz] = count_class[clazz] + 1

            #export patch to train dataset
            #logger.info('export x, y to dataset. count=' + str(count))
            pw = round(image_dims[1]/2)
            ph = image_dims[1] - pw
            trainX = utils.crop_image_fill(image_raw, (center[1]-pw,center[0]-pw), (center[1]+ph,center[0]+ph))
            target_x_ds.resize((count, image_dims[0], image_dims[1], image_dims[2]))
            target_x_ds[count-1:count] = trainX
            
            trainY = to_categorical([clazz], nb_classes=5)
            target_y_ds.resize((count, 5))
            target_y_ds[count-1:count] = trainY
            
            if(debug):
                images.append(trainX)
                cv2.circle(debug_image,center,round(w/2),(0,0,255),1)

    if(debug):
        utils.show_image(debug_image, size=8, is_bgr=True)
        utils.show_images(images, cols=12, is_bgr=True, size=1.3)
        logger.info('total animals found: ' + str(count))
    
    return count_class