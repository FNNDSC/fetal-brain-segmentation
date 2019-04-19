import os
import sys
import glob
import numpy as np
import nibabel as nib
from sklearn.model_selection import KFold

ROOT_DIR = os.path.abspath('../../../')
sys.path.append(ROOT_DIR)


def normalize_0_255(img_slice):
    limit = 0.97
    img_slice[img_slice < 0] = 0
    flat_sorted = np.sort(img_slice.flatten())

    top_limit = int(len(flat_sorted) * limit)
    intensity_limit = flat_sorted[top_limit]

    img_slice[img_slice > intensity_limit] = intensity_limit

    rows, cols = img_slice.shape
    new_img_slice = np.zeros((rows, cols))
    max_val = np.max(img_slice)

    for i in range(rows):
        for j in range(cols):
            new_img_slice[i,j] = int((
                float(img_slice[i,j]) / float(max_val)) * 255)

    return new_img_slice

def load_data_files(dataset_dir):
    images_dir = os.path.join(ROOT_DIR, dataset_dir + 'images/*')
    masks_dir = os.path.join(ROOT_DIR, dataset_dir + 'masks/*')

    image_files = sorted(glob.glob(images_dir))
    mask_files = sorted(glob.glob(masks_dir))

    return image_files, mask_files

def getKFolds(X, y, n=10, shuffle=True, seed=7):
    skf = KFold(n_splits=n, shuffle=shuffle, random_state=seed)
    skf.get_n_splits(X, y)
    return skf


