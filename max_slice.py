imgs_train_dir = 'data/train/images/*'
imgs_test_dir = 'data/test/images/*'
masks_train_dir = 'data/train/masks/*'
masks_test_dir = 'data/test/masks/*'

import os
import glob
import tqdm
import nibabel as nib
import numpy as np

def getMax(directory):
    ls = []
    glob_files = glob.glob(directory)
    max_slice = 0
    for i in tqdm.trange(len(glob_files)):
        img = (nib.load(glob_files[i])).get_fdata()
        ls.append(np.array(img))
        if i > 1:
            break
    return np.array(ls)


print(getMax(imgs_train_dir).shape)
# print(getMax(imgs_test_dir))
# print(getMax(masks_train_dir))
# print(getMax(masks_test_dir))

# print(len(ls))

# from itertools import groupby

# counts = [(i, len(list(c))) for i,c in groupby(sorted(ls))]


