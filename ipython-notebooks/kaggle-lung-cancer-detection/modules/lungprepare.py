import numpy as np
from modules.utils import Timer
from modules.logging import logger
import modules.utils as utils
import os
import dicom
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import scipy.ndimage as ndimage
from scipy.ndimage.interpolation import rotate
from scipy.ndimage.interpolation import shift
import itertools
from itertools import product, combinations
from skimage import measure, morphology, transform
import math

def find_next_valid(line, bgs=[]):
    for e in line:
        if(e not in bgs):
            return e
    return 0

# Load the scans in given folder path
#image pixels dimensions: z, y, x
def load_scan(path):
    t = Timer('load_scan ' + path)
    
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.slice_thickness = slice_thickness

    t.stop()
    return slices

#image pixels dimensions: z, y, x
def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

#image pixels dimensions: z, y, x
def resample(image, scan, new_spacing=[1,1,1]):
    t = Timer('resample')
    # Determine current pixel spacing
    spacing = np.array([scan[0].slice_thickness] + scan[0].PixelSpacing, dtype=np.float32)

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    t.stop()
    
    return image, new_spacing

def largest_label_volume(im, bgs=[]):
    vals, counts = np.unique(im, return_counts=True)
    for bg in bgs:
        counts = counts[vals != bg]
        vals = vals[vals != bg]
    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None
    
def segment_lung_mask(image, fill_lung_structures=True):
    t = Timer('segment_lung_mask')
    
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    
    #cleanup some small bubbles inside body before labelling
    binary_image = scipy.ndimage.morphology.grey_closing(binary_image, 3)

    labels = measure.label(binary_image)
    
    #Determine which label clusters refers to the air/space around the person body and turn it into the same cluster
    #The various corners are measured in case of volume being broken when the body is not fitted inside scan
    bgs = [0]
    si = np.shape(binary_image)
    si0 = si[0]-3
    si1 = si[1]-3
    si2 = si[2]-3
    for i in (2, si0):
        for j in (2, si1):
            for k in (2, si2):
                bgs.append(labels[i,j,k])

    #identify the body label
    s = np.array(np.shape(labels))
    body = find_next_valid(labels[int(s[0]*0.6), int(s[1]*0.5)], bgs=bgs)
    bgs.append(body)
    logger.debug('bgs' + str(bgs))

    #look inside the volume where lung structures is meant to be
    lung_label = largest_label_volume(labels[int(s[0]*0.2):int(s[0]*0.8), int(s[1]*0.25):int(s[1]*0.75), int(s[2]*0.25):int(s[2]*0.75)], bgs=bgs)
    logger.debug('lung_label' + str(lung_label))

    #remove everything that is not part of the lung
    logger.debug('remove non lung structures')
    binary_image[labels != lung_label] = 2
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bgs=[0])
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1
    logger.debug('fill_lung_structures')
    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    #dilate mask
    binary_image = scipy.ndimage.morphology.grey_dilation(binary_image, size=(10,10,10))
    t.stop()
    
    return binary_image

#returns ((x1, y1, z1), (x2, y2, z2))
def bounding_box(img):
    N = img.ndim
    out = []
    for ax in itertools.combinations(range(N), N - 1):
        nonzero = np.any(img, axis=ax)
        out.extend(np.where(nonzero)[0][[0, -1]])
    r = np.reshape(np.asarray(tuple(out)), (-1, 2)).T
    return [tuple(r[0]), tuple(r[1])]

#return bounding box center in (x,y,z)
def bounding_box_center(bounds):
    return (int(round((bounds[0][0] + (bounds[1][0]-bounds[0][0])/2))), int(round((bounds[0][1] + (bounds[1][1]-bounds[0][1])/2))), int(round((bounds[0][2] + (bounds[1][2]-bounds[0][2])/2))))


#find lungs rotation by finding minimum and maximum extremities from lung halves
def find_minmax_halfx(lung_mask, xhalf, bottom2up=True, left2right=True, slicen=220):
    xsize = np.shape(lung_mask)[2]-1
    ysize = np.shape(lung_mask)[1]-1
    im = np.swapaxes(lung_mask[slicen], 0, 1)

    if(bottom2up): mvalue = (-1,0)
    else: mvalue = (-1, ysize)
        
    if(left2right): 
        xstart = 0
        xend = xhalf
        xdir = 1
    else:
        xstart = xsize
        xend = xhalf
        xdir = -1
        
    for x in range(xstart, xend, xdir):
        for y in range(ysize):
            if(not bottom2up): yi = ysize - y
            else: yi = y

            if(im[x][yi]>0.5):
                if(bottom2up and yi>mvalue[1]):
                    mvalue = (x, yi)
                elif(not bottom2up and yi<mvalue[1]):
                    mvalue = (x, yi)
    return mvalue
    
def calculate_angle(p1, p2):
    return math.degrees(math.atan2(p2[1]-p1[1],p2[0]-p1[0]))

def value_between(value, min_value, max_value):
    if(value<min_value): return False
    if(value>max_value): return False
    return True

def discover_lung_rotation(lung_mask):
    bbox = bounding_box(lung_mask)
    if(bbox == None): return 0
    slicen = int((bbox[1][2]-bbox[0][2])/2)
    half = int(bbox[0][0]+(bbox[1][0]-bbox[0][0])/2)

    l1 = find_minmax_halfx(lung_mask, half, bottom2up=True, left2right=True, slicen=slicen)
    r1 = find_minmax_halfx(lung_mask, half, bottom2up=True, left2right=False, slicen=slicen)
    l2 = find_minmax_halfx(lung_mask, half, bottom2up=False, left2right=True, slicen=slicen)
    r2 = find_minmax_halfx(lung_mask, half, bottom2up=False, left2right=False, slicen=slicen)

    r = (l1, r1, l2, r2)
    xs, ys = zip(*r)
    
    #verify points sanity
    if(not value_between(xs[1]-xs[0], 50, 200) or
       not value_between(xs[3]-xs[2], 50, 200) or
       not value_between(ys[0]-ys[2], 100, 250) or
       not value_between(ys[1]-ys[3], 100, 250)):
        logger.warning('Strange rotation detected. returning 0 degrees')
        return 0
    
    angle1 = calculate_angle(l1, r1)
    angle2 = calculate_angle(l2, r2)
    
    a = ((angle1 + angle2)/2)
    return min(max(a, -10), 10)

def diff_for_shiffiting(point1, point2):
    t = np.subtract(point1, point2)
    return (t[2], t[1], t[0])

def bbox_dim(bbox):
    bw = bbox[1][0]-bbox[0][0]
    bh = bbox[1][1]-bbox[0][1]
    bd = bbox[1][2]-bbox[0][2]
    return bw,bh,bd


def process_patient_images(patient_dir, image_dims, output_dir, patient_id):
    patient_scan = load_scan(patient_dir)
    patient_pixels = get_pixels_hu(patient_scan)
    utils.show_slices(patient_pixels, str(patient_id) + 'hu', nr_slices=12, cols=4, output_dir=output_dir + 'images')
    patient_pixels, spacing = resample(patient_pixels, patient_scan, [1,1,1])
    utils.show_slices(patient_pixels, str(patient_id) + 'resampled', nr_slices=12, cols=4, output_dir=output_dir + 'images')
    patient_lung_mask = segment_lung_mask(patient_pixels, True)
    utils.show_slices(patient_pixels, str(patient_id) + 'mask', nr_slices=12, cols=4, output_dir=output_dir + 'images')
    
    t = Timer('apply lung mask to image volume')
    patient_pixels = np.ma.masked_where(patient_lung_mask==0, patient_pixels).filled(fill_value=0)
    utils.show_slices(patient_pixels, str(patient_id) + 'masked', nr_slices=12, cols=4, output_dir=output_dir + 'images')
    t.stop()

    t = Timer('rotate image for optimal pose ' + patient_dir)
    rotation_angle = discover_lung_rotation(patient_lung_mask)
    patient_pixels = rotate(patient_pixels,rotation_angle,(1,2), reshape=False)
    utils.show_slices(patient_pixels, str(patient_id) + 'rotated', nr_slices=12, cols=4, output_dir=output_dir + 'images')
    t.stop()
    
    t = Timer('resize image volume to {}x{}x{}'.format(image_dims[0], image_dims[1], image_dims[2]))
    bbox = bounding_box(patient_pixels)
    if(bbox == None):
        return None
    bw,bh,bd = bbox_dim(bbox)
    fit_volume = (image_dims[2], image_dims[1], image_dims[0])
    ratio = min(tuple(np.divide(fit_volume,np.subtract(bbox[1],bbox[0]))))
    logger.debug('ratio=' + str(ratio))
   
    patient_pixels = scipy.ndimage.interpolation.zoom(patient_pixels[bbox[0][2]:bbox[1][2],bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0]], ratio)
    t.stop()

    t = Timer('translate to center')
    fit_volume_center = tuple(np.divide(fit_volume, 2))
    bbox = bounding_box(patient_pixels)
    bbox_center = bounding_box_center(bbox)

    patient_pixels2 = np.full((image_dims[0], image_dims[1], image_dims[2]),0)
    ps = np.shape(patient_pixels)
    patient_pixels2[:ps[0],:ps[1],:ps[2]] = patient_pixels[:ps[0],:ps[1],:ps[2]]
    patient_pixels = patient_pixels2
    
    diff = (np.subtract(fit_volume_center,bbox_center))
    patient_pixels = shift(patient_pixels, (diff[2],diff[1],diff[0]))
    t.stop()

    #normalization for better training on neural networks
    t = Timer('pixel normalization')
    MIN_BOUND = -1000.0
    MAX_BOUND = 400.0
    
    patient_pixels = (patient_pixels - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    patient_pixels[patient_pixels>1] = 1.
    patient_pixels[patient_pixels<0] = 0.

    #0-center pixels
    logger.debug('mean pixels=' + str(np.mean(patient_pixels)))
    PIXEL_MEAN = 0.6 #calculated before
    patient_pixels = patient_pixels - PIXEL_MEAN
    t.stop()
    
    #add color channel dimension
    patient_pixels = np.expand_dims(patient_pixels, axis=3)
    
    return patient_pixels

