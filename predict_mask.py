import numpy as np

import os
import skimage.io as io
from datahandler import DataHandler
from unet import getUnet
from medpy.io import save
from keras import backend as K

# get keras model
model = getUnet()
model.load_weights('unet_brain_seg_axial.h5')

# get image data and header
dh = DataHandler()
image, hdr = dh.getImageData('./data/prev/04.nii')
save_path = './data/prev/'

# predict mask
results = model.predict(image, verbose = 1)

# save mask to nifti format
# remove extra axis
n_results = np.squeeze(results)
# swap axis so they are (size, size, num slices)
n_results = np.moveaxis(n_results, 0, -1)
save(n_results, os.path.join(save_path, 'mask_04.nii'), hdr)
