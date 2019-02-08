import tensorflow as tf
import numpy as np

from keras.models import *
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from unet import *
from datahandler import DataHandler
import os
import skimage.io as io
from tqdm import tqdm
from math import ceil
import glob
from medpy.io import save

model = getUnet()
model.load_weights('unet_brain_seg.h5')

dh = DataHandler()

def destiny_directory(dice_score):
    pre = './data/eval/'
    if dice_score >= 98:
        return pre + 'dice_98_100/'
    elif dice_score >= 96:
        return pre + 'dice_96_98/'
    elif dice_score >= 94:
        return pre + 'dice_94_96/'
    elif dice_score >= 92:
        return pre + 'dice_92_94/'
    elif dice_score >= 90:
        return pre + 'dice_90_92/'
    elif dice_score >= 88:
        return pre + 'dice_88_90/'
    elif dice_score >= 85:
        return pre + 'dice_85_88'
    elif dice_score >= 80:
        return pre + 'dice_80_85/'
    elif dice_score >= 70:
        return pre + 'dice_70_80/'
    elif dice_score >= 60:
        return pre + 'dice_60_70/'
    else:
        return pre + 'dice_less_60'

def getFileName(fname):
    original_name = fname.split('/')[-1]
    original_name = original_name[:original_name.index('.')]
    return original_name

def predict_masks(fname, image, o_image, o_mask, hdr, dice_score):
    # get image data and header
    dice_score = int(dice_score * 100)
    save_path = destiny_directory(dice_score)

    # predict mask
    results = model.predict(image, verbose = 1)

    # save mask to nifti format
    # remove extra axis
    n_results = np.squeeze(results)
    # swap axis so they are (size, size, num slices)
    n_results = np.moveaxis(n_results, 0, -1)

    new_name = fname + '_pred_dice_' + str(dice_score)

    save(n_results, os.path.join(save_path, new_name + '.nii'), hdr)
    save(o_image, os.path.join(save_path, fname + '_img.nii'), hdr)
    save(o_mask, os.path.join(save_path, fname + '_mask.nii'), hdr)

def eval_prediction(image, mask):
    return  model.evaluate(image, mask, verbose = 1)

images_dir = './data/test/images/*'
masks_dir = './data/test/masks/*'

glob_images = sorted(glob.glob(images_dir))
glob_masks = sorted(glob.glob(masks_dir))

for file_image, file_mask in zip(glob_images, glob_masks):
    original_img, img, hdr = dh.getImageData(file_image,
            is_eval = True)
    original_msk, msk, _ = dh.getImageData(file_mask,
            is_mask = True,
            is_eval = True)

    res = eval_prediction(img, msk)

    fname = getFileName(file_image)

    predict_masks(fname, img, original_img,
            original_msk, hdr, res[1])
    print(res)

