import glob
import SimpleITK as sitk
import numpy as np

class CTScanMhd(object):
    def __init__(self, base_dir, filename):
        self.filename = filename
        self.coords = None
        self.base_dir = base_dir
        path = glob.glob(self.base_dir + '/*/' + self.filename + '.mhd')
        self.ds = sitk.ReadImage(path[0])
        self.image = sitk.GetArrayFromImage(self.ds)

    def set_coords(self, coords):
        self.coords = (coords[2], coords[1], coords[0])

    def get_resolution(self):
        return self.ds.GetSpacing()

    def get_origin(self):
        return self.ds.GetOrigin()

    def get_ds(self):
        return self.ds

    def get_voxel_coords(self):
        origin = self.get_origin()
        resolution = self.get_resolution()
        voxel_coords = [np.absolute(self.coords[j]-origin[j])/resolution[j] \
            for j in range(len(self.coords))]
        return tuple(voxel_coords)
    
    def get_image(self):
        return self.image
    
    def get_subimage(self, center, dims):
        self.set_coords(center)
        x, y, z = self.get_voxel_coords()
        subImage = self.image[int(z-dims[0]/2):int(z+dims[0]/2), int(y-dims[1]/2):int(y+dims[1]/2), int(x-dims[2]/2):int(x+dims[2]/2)]
        return subImage
    
    def get_normalized_image(self, minHU, maxHU):
        #maxHU = 400.
        #minHU = -1000.
        img = (self.image - minHU) / (maxHU - minHU)
        img[img>1] = 1.
        img[img<0] = 0.
        return img
  