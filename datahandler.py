import os
import glob
import tqdm
import cv2

import numpy as np
import nibabel as nib
from medpy.io import load

class DataHandler:
    def __init__(self):
        self._imgs_train_dir = 'data/train/images/*'
        self._imgs_test_dir = 'data/test/images/*'

        self._masks_train_dir = 'data/train/masks/*'
        self._masks_test_dir = 'data/test/masks/*'

    # values must be between 0 and 255
    def __normalize0_255(self, img_slice):
        img_slice[img_slice < 0] = 0
        flat_sorted = np.sort(img_slice.flatten())

        top_3_limit = int(len(flat_sorted) * 0.97)
        limit = flat_sorted[top_3_limit]

        img_slice[img_slice > limit] = limit


        rows, cols = img_slice.shape
        new_img = np.zeros((rows, cols))
        max_val = np.max(img_slice)

        for i in range(rows):
            for j in range(rows):
                new_img[i,j] = int((float(img_slice[i,j])/float(max_val)) * 255)

        # new_img = (new_img - new_img.mean()) / new_img.std()

        return new_img

    # Receives array of file names and
    # corresponding index
    # returns image instance
    def __getImage(self, glob_name):
        img = nib.load(glob_name)
        return img.get_fdata()

    # receives directories for both
    # directories containing images and masks
    # returns a tuple with two arrays for
    # all images and masks
    def __getImages(self, images_dir=None, masks_dir=None,
            desc='Data aquisition', image_names=None,
            mask_names=None, from_names=False):

        images, masks = [], []

        #sorted as to insure img/msk match

        if not from_names:
            image_names = sorted(glob.glob(images_dir))
            mask_names = sorted(glob.glob(masks_dir))

        for file_i in tqdm.trange(len(image_names),
                ncols = 80, desc = desc):

            img_name = image_names[file_i]
            mask_name = mask_names[file_i]
            # get all slices of nifti image
            img_slices = self.__getImage(img_name)
            mask_slices = self.__getImage(mask_name)
            # itirate through each slice
            for i in range(img_slices.shape[-1]):
                img = np.array(img_slices[:,:,i])
                mask = np.array(mask_slices[:,:,i])

                #resize images into 256x256
                if img.shape[0] != 256 or img.shape[1] != 256:
                    img = cv2.resize(img, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)
                    mask = cv2.resize(mask, dsize=(256, 256), interpolation=cv2.INTER_CUBIC)

                # Normalize image so its between 0-255
                new_img = self.__normalize0_255(img)

                # add new channel axis to img and mask
                img = new_img[..., np.newaxis]
                mask[mask > 1] = 1
                mask[mask < 1] = 0
                mask = mask[..., np.newaxis]
                images.append(img)
                masks.append(mask * 255)
        images = np.array(images, dtype=np.uint16)
        masks = np.array(masks, dtype=np.uint16)
        return (images, masks)

    def getFilesFromIndices(self, files, indices):
        return np.take(files, indices)

    def getKFoldData(self, image_files, mask_files, indices, eval=False):

        image_train_files = self.getFilesFromIndices(image_files, indices['train'])
        mask_train_files = self.getFilesFromIndices(mask_files, indices['train'])

        tr_images, tr_masks = self.__getImages(image_names = image_train_files,
                mask_names = mask_train_files, from_names=True)

        image_val_files = self.getFilesFromIndices(image_files, indices['val'])
        mask_val_files = self.getFilesFromIndices(mask_files, indices['val'])

        val_images, val_masks = self.__getImages(image_names = image_val_files,
                mask_names = mask_val_files, from_names=True)

        if eval:
            return tr_images, tr_masks

        return tr_images, tr_masks, val_images, val_masks

    # return tuple containing training data
    def getTrainData(self):
        return self. __getImages(self._imgs_train_dir,
                self._masks_train_dir,
                desc = 'Training data')

    # return tuple containing testing data
    def getTestData(self):
        return self.__getImages(self._imgs_test_dir,
                self._masks_test_dir,
                desc = 'Validation data')

    #return image data
    def getImageData(self, fname, is_mask=False):
        # get image data and header, must use med.py
        # for internal process of getting header info
        data, hdr = load(fname)

        # switch axis
        data = np.moveaxis(data, -1, 0)

        if is_mask:
            data = np.array(data, dtype=np.uint16)
            data = data[..., np.newaxis]
            return data, hdr

        norm_data = []

        # normalize each slice
        for i in range(data.shape[0]):
            img_slice = data[i,:,:]
            norm_data.append(self.__normalize0_255(img_slice))

        #change to numpy array and add axis
        data = np.array(norm_data, dtype=np.uint16)
        data = data[..., np.newaxis]
        return data, hdr

    # return arrays containing all training and test data
    # as single 2D slices
    # when only_test active returns only the testing images
    def getData(self, only_test = False):
        if not only_test:
            tr_images, tr_masks = self.getTrainData()
            te_images, te_masks = self.getTestData()
            return tr_images, tr_masks, te_images, te_masks

        else:
            te_images, te_masks = self.getTestData()
            return (te_images, te_masks)
