import numpy as np
import skimage.feature as feature
import skimage.transform as transform
from multiprocessing import Pool
import multiprocessing
import itertools

from modules.logging import logger
import modules.utils as utils
from modules.utils import Timer

def sliding_window_generator(image, step=(15,15), window=(32,32), pyramid_scale=0.5, pyramid_max_layers=1):
    """
       Generator of slices over the entire image in 2D
       If pyramid_downscale!=0, will generate slices on reduced image size (maintaining slice size) 
       in layers (pyramid), until image size is less than window size (first layer is not reduced)
       image: 2D image
       step: x,y step sizes
       window: w,h of the sliding window size
       pyramid_ratio: reduction factor at each reduction iteration
       returns: image slice data generator for items in the format (x1, y1, x2, y2, pyramid_scale)
    """
#     multichannel = (len(image.shape)==3)
    # iterate over image layers
#    utils.show_image(image)
#     for im_scaled in transform.pyramid_gaussian(image, downscale=pyramid_downscale, max_layer=pyramid_max_layers):#, multichannel=multichannel):
    for im_scaled,scale in pyramid_generator(image, scale=pyramid_scale, max_layers=pyramid_max_layers):#, multichannel=multichannel):
#        utils.show_image(im_scaled)
        if im_scaled.shape[1] < window[1] or im_scaled.shape[1] < window[1]:
            return
        t = Timer('sliding_window')
        # slide a window across the image
        for y in range(0, im_scaled.shape[0], step[0]):
            utils.print_progress(y, im_scaled.shape[0], elapsed_seconds=t.elapsed(), status='sliding window')
            for x in range(0, im_scaled.shape[1], step[1]):
                # yield the current window
                yield (y, x, im_scaled[y:y + window[0], x:x + window[1]], scale)
        t.stop()

def evaluate_regions(region_generator, evaluate_function, score_threshold=10, apply_non_max_suppression=True, supression_overlap_threshold=0.3, threads=None, batch_size=100):
    """
       Iterate over region generator and for each region, call evaluate_function.
       image=2D greyscale image
       step=x,y step sizes
       window=w,h of the sliding window size
       score_threshold=minimum score for evaluated region that will be return in results
       returns: detected_boxes, boxes_images
                ex: box1 = (x1,y1,x2,y2,score)
                    box2 = (x1,y1,x2,y2,score)
                    return [box1,box2], [img1,img2]
    """
    if threads==None or threads<=0:
        threads = multiprocessing.cpu_count()
        
    detections = []
    images = []
    
    with Pool(threads) as p:
        er = EvalRegion()
        #process in batches
        batch_regions = [0]
        while(len(batch_regions)>0):
            batch_regions = [(r,evaluate_function,score_threshold) for r in itertools.islice(region_generator, batch_size)]
            dets_imgs = p.starmap(er.evaluate_region, batch_regions)
            dets_imgs = [x for x in dets_imgs if x[0] is not None]
            dets_imgs = np.array(dets_imgs)
            if(dets_imgs.shape[0]>0):
                detections += (dets_imgs[:,0].tolist())
                images += (dets_imgs[:,1].tolist())
            
        if apply_non_max_suppression:
            t = Timer('non_max_suppression')
            detections = non_maxima_suppression(np.array(detections), overlap_threshold=supression_overlap_threshold)
            t.stop()
            
        return np.array(detections, dtype='int'), images

class EvalRegion():
    win_size = None
    def evaluate_region(self, region, evaluate_function, score_threshold):
        img = region[2]
        if self.win_size==None:
            self.win_size = img.shape
        #boundary patches are smaller than the first ones
        if img.shape[0]==self.win_size[0] and img.shape[1]==self.win_size[1]:
            score = evaluate_function(img)
            scale = region[3]
            eval_detection = [region[0], region[1], img.shape[0]*(1/scale), img.shape[1]*(1/scale), score]
            if score>=score_threshold:
                return eval_detection, img
        return None, None

def pyramid_generator(image, scale=0.5, max_layers=-1):
    current_scale = 1
    if(max_layers==-1):
        max_layers = 99999
    for layer in range(max_layers):
        if(layer>0):
            downscale = int(1/scale)
            image = transform.downscale_local_mean(image, (downscale,downscale))
            current_scale = current_scale*scale
            if image.shape[0]==1 or image.shape[1]==1:
                return
        yield image, current_scale

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


def non_maxima_suppression(detections, overlap_threshold=0.2):
    '''
    This function performs Non-Maxima Suppression.
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
    new_detections=[]
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
