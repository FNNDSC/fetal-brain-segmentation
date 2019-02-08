import os
import glob
import tqdm

import numpy as np
import nibabel as nib
from medpy.io import load

class DataHandler:


    def __init__(self):
        self._imgs_train_dir = 'data/train/axial_only/images/*'
        self._imgs_test_dir = 'data/test/axial_only/images/*'

        self._masks_train_dir = 'data/train/axial_only/masks/*'
        self._masks_test_dir = 'data/test/axial_only/masks/*'

    # values must be between 0 and 255
    def __normalize0_255(self, img_slice):
        rows, cols = img_slice.shape
        new_img = np.zeros((rows, cols))
        max_val = np.max(img_slice)

        for i in range(rows):
            for j in range(rows):
                new_img[i,j] = int((float(img_slice[i,j])/float(max_val)) * 255)

        return new_img

    # Receives array of file names and
    # corresponding index
    # returns image instance
    def __getImage(self, glob_name, i):
        img = nib.load(glob_name[i])
        return img.get_fdata()

    # receives directories for both
    # directories containing images and masks
    # returns a tuple with two arrays for
    # all images and masks
    def __getImages(self, images_dir, masks_dir,
            desc = 'Data aquisition', has_3_channels = False):
        images, masks = [], []

        #sorted as to insure img/msk match
        glob_images = sorted(glob.glob(images_dir))
        glob_masks = sorted(glob.glob(masks_dir))

        for glob_i in tqdm.trange(len(glob_images),
                ncols = 80, desc = desc):

            # get all slices of nifti image
            img_slices = self.__getImage(glob_images, glob_i)
            mask_slices = self.__getImage(glob_masks, glob_i)

            # itirate through each slice
            # TODO limit range of values between 0-1
            for i in range(img_slices.shape[-1]):
                img = np.array(img_slices[:,:,i])
                mask = np.array(mask_slices[:,:,i])

                # skip images that are not 256x256
                # TODO Resize them
                if img.shape[0] != 256 or img.shape[1] != 256:
                    break

                # Normalize image so its between 0-255
                new_img = self.__normalize0_255(img)

                # add new channel axis to img and mask
                img = new_img[..., np.newaxis]
                mask = mask[..., np.newaxis]

                if has_3_channels:
                    images.append(np.array(np.dstack((img,img,img))))
                else:
                    images.append(img)
                masks.append(mask)

        images = np.array(images, dtype=np.uint16)
        masks = np.array(masks, dtype=np.uint16)

        return (images, masks * 255)

    # return tuple containing training data
    def getTrainData(self, has_3_channels = False):
        return self. __getImages(self._imgs_train_dir,
                self._masks_train_dir,
                desc = 'Training data', has_3_channels = has_3_channels)

    # return tuple containing testing data
    def getTestData(self, has_3_channels = False):
        return self.__getImages(self._imgs_test_dir,
                self._masks_test_dir,
                desc = 'Validation data', has_3_channels = has_3_channels)

    #return image data
    def getImageData(self, fname):
        # get image data and header, must use med.py
        # for internal process of getting header info
        data, hdr = load(fname)

        # switch axis
        data = np.moveaxis(data, -1, 0)
        norm_data = []

        # normalize each slice
        for i in range(data.shape[0]):
            img_slice = data[i,:,:]
            norm_data.append(self.__normalize0_255(img_slice))

        #change to numpy array and add axis
        norm_data = np.array(norm_data, dtype=np.uint16)
        data = data[..., np.newaxis]
        return data, hdr

    # return arrays containing all training and test data
    # as single 2D slices
    # when only_test active returns only the testing images
    def getData(self,
            only_test = False,
            has_3_channels = False):

        if not only_test:
            tr_images, tr_masks = self.getTrainData(has_3_channels = has_3_channels)
            te_images, te_masks = self.getTestData(has_3_channels = has_3_channels)
            return tr_images, tr_masks, te_images, te_masks

        else:
            te_images, te_masks = self.getTestData(has_3_channels = has_3_channels)
            return (te_images, te_masks)
