import numpy as np
import skimage.feature as feature
import skimage.transform as transform
from multiprocessing import Pool
import multiprocessing
import itertools
import cv2
import tensorflow as tf

from modules.logging import logger
import modules.utils as utils
from modules.utils import Timer

def sliding_window_generator(image, step=(15,15), window=(32,32), pyramid_firstscale=None, pyramid_scale=0.5, pyramid_max_layers=1):
    """
       Generator of slices over the entire image in 2D
       If pyramid_downscale!=0, will generate slices on reduced image size (maintaining slice size) 
       in layers (pyramid), until image size is less than window size (first layer is not reduced)
       image: 2D image
       step: x,y step sizes
       window: w,h of the sliding window size
       pyramid_ratio: reduction factor at each reduction iteration
       returns: image slice data generator for items in the format (y, x, image_region_scaled, pyramid_scale)
    """
    
    print(image.shape)
    
    if(pyramid_firstscale!=None):
        image = transform.rescale(image, pyramid_firstscale)
        print(image.shape)
    
    for im_scaled,scale in pyramid_generator(image, scale=pyramid_scale, max_layers=pyramid_max_layers):
        if im_scaled.shape[1] < window[1] or im_scaled.shape[1] < window[1]:
            return
        t = Timer('sliding_window')
        # slide a window across the image
        for y in range(0, im_scaled.shape[0], step[0]):
            utils.print_progress(y, im_scaled.shape[0], elapsed_seconds=t.elapsed(), status='sliding window', show_remaining=True)
            for x in range(0, im_scaled.shape[1], step[1]):
                # yield the current window
                yield (y, x, im_scaled[y:y + window[0], x:x + window[1]], scale)
        t.stop()

def evaluate_regions(region_generator, evaluate_function, filter_score_min=0.7, filter_labels=None, apply_non_max_suppression=True, supression_overlap_threshold=0.3, threads=None, batch_size=100, apply_nms_each=3000000):
    """
       Iterate over region generator and for each region, call evaluate_function.
       image=2D greyscale image
       step=x,y step sizes
       window=w,h of the sliding window size
       score_threshold=minimum score for evaluated region that will be return in results
       threads: if None, no multithreading applied. if -1, uses maximum nr of cores or a specific number of threads
       apply_nms_each: if you are experiencing too much memory overhead due to a high number of detections, set this parameter to something in order of millions so that intermediary cleanups (nms) will be performed during evaluation. Note that a lower number may reduce a little the quality of NMS because it ideally uses the entire dataset to analyse suppression.
       returns: detections, patches
                ex: detection1 = (x1,y1,x2,y2,score,label,text)
                    detection2 = (x1,y1,x2,y2,score,label,text)
                    return [detection1,detection2], [patch1,patch2]
    """
        
    detections = []
    images = []

    #NOT USING THREADS
    if(threads==None):
        er = EvalRegion()
        dets_imgs = []
        for region in region_generator:
            ei = er.evaluate_region(region, evaluate_function, filter_score_min, filter_labels)
            if(ei[0] is not None):
                dets_imgs.append(ei)
            dets_imgs0 = np.array(dets_imgs)
            if(dets_imgs0.shape[0]>0):
                detections += (dets_imgs0[:,0].tolist())
                images += (dets_imgs0[:,1].tolist())
            #cleanup overlapping boxes from time to time to avoid too much memory usage
            if(apply_non_max_suppression and len(detections)>3000000):
                t = Timer('non_max_suppression. boxes=' + str(len(detections)))
                detections = non_maxima_suppression(detections, overlap_threshold=supression_overlap_threshold)
                t.stop()

    
    #USING THREADS
    else:
        if threads<=0:
            threads = multiprocessing.cpu_count()
        with Pool(threads) as p:
            er = EvalRegion()
            #process in batches
            batch_regions = [0]
            while(len(batch_regions)>0):
                batch_regions = [(r,evaluate_function,filter_score_min,filter_labels) for r in itertools.islice(region_generator, batch_size)]
                dets_imgs = p.starmap(er.evaluate_region, batch_regions)
                dets_imgs = [x for x in dets_imgs if x[0] is not None]
                dets_imgs = np.array(dets_imgs)
                if(dets_imgs.shape[0]>0):
                    detections += (dets_imgs[:,0].tolist())
                    images += (dets_imgs[:,1].tolist())

    #TODO: APPLY NNS AT EACH X DETECTIONS, INSTEAD TO LET IT TO THE END SO THAT THE ALGO GETS OPTMIZED (MAYBE?)
    #4869104
    if apply_non_max_suppression:
        t = Timer('non_max_suppression. boxes=' + str(len(detections)))
        detections = non_maxima_suppression(detections, overlap_threshold=supression_overlap_threshold)
        t.stop()

    return np.array(detections), images

class EvalRegion():
    win_size = None
    def evaluate_region(self, region, evaluate_function, filter_score_min, filter_labels):
        img = region[2]
        if self.win_size==None:
            self.win_size = img.shape
        #boundary patches are smaller than the first ones
        if img.shape[0]==self.win_size[0] and img.shape[1]==self.win_size[1]:
            score,label = evaluate_function(img)
            scale = region[3]
            eval_detection = [region[0], region[1], round(img.shape[0]*(1/scale)), round(img.shape[1]*(1/scale)), score, label]
            if score>=filter_score_min and (filter_labels==None or label in filter_labels):
                return eval_detection, img
        return None, None

def pyramid_generator(image, scale=0.5, max_layers=-1):
    current_scale = 1
    if(max_layers==-1):
        max_layers = 99999
    for layer in range(max_layers):
        if(layer>0):
#            if(len(image.shape)==2):
#                downscale = (downscale,downscale)
#            elif(len(image.shape)==3):
#                downscale = (downscale,downscale,1)
            print(str(image.shape) + ' ' + str(scale))
#            image = transform.downscale_local_mean(image, downscale)
            image = transform.rescale(image, scale)
            print(str(image.shape))
            current_scale = current_scale*scale
            if image.shape[0]==1 or image.shape[1]==1:
                return
        logger.info('pyramid layer=' + str(layer) + ' image=' + str(image.shape) + ' scale=' + str(current_scale))
        yield image, current_scale

def non_maxima_suppression(detections, overlap_threshold=0.2, max_detections=99999, strict=True):
    """
    Performs Non Maxima Suppresion algorithm removing overlapping boxes on detections list
    detections: list of detections in format (y,x,h,w,score)
    overlap_threshold: percentage of boxes with overlapping areas to keep
    max_detections: max boxes in result
    strict: Tensor Flow NMS implementation used by this function is very optimized, but it keep some boxes when
            box sizes differ. Setting this to true will apply a more precise (but slower) algorithm on results
            before returning in order to remove those remaining boxes.
    """
    if(len(detections)==0):
        return detections
    detections = np.array(detections)
    b = detections[:,0:4]
    #tensorflow need boxes in y1,x1,y2,x2 (we have boxes in y,x,h,w)
    boxes = np.array([(c[0],c[1],c[0]-c[2],c[1]-c[3]) for c in b], dtype='int')
    scores = detections[:,4]
    keeped_detections = tf.image.non_max_suppression(
        boxes,
        scores,
        max_detections,
        iou_threshold=overlap_threshold,
        name=None
    )
    with tf.Session() as sess:
        detections = detections[keeped_detections.eval()]
    
    if(strict):
        detections = non_maxima_suppression_precise(detections, overlap_threshold=overlap_threshold)

    return detections
    
    
def overlapping_area(detection_1, detection_2):
    '''
    Function to calculate overlapping area'si
    `detection_1` and `detection_2` are 2 detections whose area
    of overlap needs to be found out.
    Each detection is list in the format ->
    [y-top-left, x-top-left, height-of-detection, width-of-detection, confidence-of-detections]
    The function returns a value between 0 and 1,
    which represents the area of overlap.
    0 is no overlap and 1 is complete overlap.
    Area calculated from ->
    http://math.stackexchange.com/questions/99565/simplest-way-to-calculate-the-intersect-area-of-two-rectangles
    '''
    # Calculate the x-y co-ordinates of the 
    # rectangles
    x1_tl = detection_1[0]
    x2_tl = detection_2[0]
    x1_br = detection_1[0] + detection_1[2]
    x2_br = detection_2[0] + detection_2[2]
    y1_tl = detection_1[1]
    y2_tl = detection_2[1]
    y1_br = detection_1[1] + detection_1[3]
    y2_br = detection_2[1] + detection_2[3]
    # Calculate the overlapping Area
    x_overlap = max(0, min(x1_br, x2_br)-max(x1_tl, x2_tl))
    y_overlap = max(0, min(y1_br, y2_br)-max(y1_tl, y2_tl))
    overlap_area = x_overlap * y_overlap
    area_1 = detection_1[2] * detection_2[3]
    area_2 = detection_2[2] * detection_2[3]
    total_area = area_1 + area_2 - overlap_area
    return overlap_area / float(total_area)


def non_maxima_suppression_precise(detections, overlap_threshold=0.2):
    '''
    This function performs Non-Maxima Suppression. Tensor Flow implementation is way more optimized than this (this
    implementation doesn't scale at millions of samples), but it keeps some boxes when they have different sizes. 
    After passing TF implementation, using this algorithm on result will cleanup remanining boxes.
    `detections` consists of a list of detections.
    Each detection is in the format ->
    [y-top-left, x-top-left, height-of-detection, width-of-detection, confidence-of-detections]
    If the area of overlap is greater than the `threshold`,
    the area with the lower confidence score is removed.
    The output is a list of detections.
    Returns detections
    '''
    if len(detections) == 0:
        return []
    # Sort the detections based on confidence score
    detections = sorted(detections, key=lambda detections: detections[4], reverse=True)
    # Unique detections will be appended to this list
    new_detections = []
    # Append the first detection
    new_detections.append(detections[0])
    # Remove the detection from the original list
    del detections[0]
    # For each detection, calculate the overlapping area
    # and if area of overlap is less than the threshold set
    # for the detections in `new_detections`, append the 
    # detection to `new_detections`.
    # In either case, remove the detection from `detections` list.
    for index, detection in enumerate(detections):
        for new_detection in new_detections:
            if overlapping_area(detection, new_detection) > overlap_threshold:
                del detections[index]
                break
        else:
            new_detections.append(detection)
            del detections[index]

    return new_detections    
    
    
def draw_detections(detections, img, detection_to_colortext=None):
    """ detections is an array of (x0, y0, x1, y1, score, label) """
    for i, detection in enumerate(detections):
        detint = detection.astype('int')
        score = int(detections[i][4])
        p = img[detint[0]:detint[0]+detint[2],detint[1]:detint[1]+detint[3]]
        color = (0,0,255)
        if(detection_to_colortext!=None):
            color,text = detection_to_colortext(detection)
            if(color is None):
                color = (0,0,255)
            cv2.rectangle(img, (detint[1],detint[0]), (detint[3]+detint[1],detint[2]+detint[0]), color=color, thickness=2)
            if(text is not None):
                cv2.putText(img,text,(detint[1]+5,detint[0]+18), cv2.FONT_HERSHEY_SIMPLEX, 0.3,(0,0,255),1,cv2.LINE_AA)

def extract_patches(detections, img):
    patches = []
    for i, detection in enumerate(detections):
        region = detection.astype('int')
        p = img[region[0]:region[0]+region[2],region[1]:region[1]+region[3]]
        patches.append(p)
    return patches
