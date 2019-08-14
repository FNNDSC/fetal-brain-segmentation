import os
import sys
import glob
import numpy as np
import nibabel as nib
from sklearn.model_selection import KFold

ROOT_DIR = os.path.abspath('.')
sys.path.append(ROOT_DIR)

#Function to get all the data files available, as strings
# this is just to get the array of file names
# sorting ensures that mask and img correspond to eachother
def load_data_files(dataset_dir):
    images_dir = os.path.join(ROOT_DIR, dataset_dir + 'images/*')
    masks_dir = os.path.join(ROOT_DIR, dataset_dir + 'masks/*')

    image_files = sorted(glob.glob(images_dir))
    mask_files = sorted(glob.glob(masks_dir))

    return image_files, mask_files

#get scikit kfold cross validation indices, seed ensures they can be replicated
def getKFolds(X, y, n=10, shuffle=True, seed=7):
    skf = KFold(n_splits=n, shuffle=shuffle, random_state=seed)
    skf.get_n_splits(X, y)
    return skf


