import os
import glob
import tqdm

import numpy as np
import nibabel as nib

class DataHandler:


    def __init__(self):
        self._imgs_train_dir = 'data/train/images/*'
        self._imgs_test_dir = 'data/test/images/*'

        self._masks_train_dir = 'data/train/masks/*'
        self._masks_test_dir = 'data/test/masks/*'

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
            desc = 'Data aquisition'):
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

                # add new channel axis to img and mask
                img = img[..., np.newaxis]
                mask = mask[..., np.newaxis]

                images.append(img)
                masks.append(mask)

        images = np.array(images, dtype=np.uint8)
        masks = np.array(masks, dtype=np.uint8)

        return (images, masks * 255)

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

    # return arrays containing all training and test data
    # as single 2D slices
    # when only_test active returns only the testing images
    def getData(self, only_test =
            False):
        if not only_test:
            tr_images, tr_masks = self.getTrainData()
            te_images, te_masks = self.getTestData()

            return tr_images, tr_masks, te_images, te_masks

        else:
            te_images, _ = self.getTestData()
            return te_images
