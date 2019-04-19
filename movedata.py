import os
import glob
import numpy as np

img_dir = 'data/Brain_mask/images/*'
mask_dir = 'data/Brain_mask/masks/*'

data = zip(sorted(glob.glob(img_dir)),
        sorted(glob.glob(mask_dir)))
i = 215
for img, msk in data:
    new_img = img[:img.rindex('/')+1] + '%d.nii'%i
    new_msk = msk[:msk.rindex('/')+1] + '%d.nii'%i
    os.rename(img, new_img)
    os.rename(msk, new_msk)
    i += 1
