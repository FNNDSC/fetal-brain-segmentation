import numpy as np

from keras.models import *
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

from models.unet import *
from datahandler import DataHandler

import os
import skimage.io as io
from tqdm import tqdm
from math import ceil
# TODO remove this
import warnings
warnings.filterwarnings("ignore")

model = getUnet()
model.load_weights('logs/unet/unet_dice_nobells/unet_dice_nobells_weights.h5')

dh = DataHandler()
images, masks = dh.getData(only_test = True)

# save_path = './data/test_results/'
# for i, img in enumerate(tqdm(images, desc='Saving Imgs')):
#     io.imsave(os.path.join(save_path,"%d_img.png"%i), np.squeeze(img))
#     io.imsave(os.path.join(save_path,"%d_mask.png"%i), np.squeeze(masks[i]))

def resetSeed():
    np.random.seed(1)

def getGenerator(images):
    resetSeed()

    image_datagen = ImageDataGenerator(rescale=1./255)
    image_datagen.fit(images, augment = True)
    image_generator = image_datagen.flow(x = images,
            shuffle = False)

    return image_generator


def dice_coef(y_true, y_pred, smooth=1):
    y_true = np.logical_not(y_true.flatten().astype(np.bool))
    y_pred = y_pred.flatten().astype(np.bool)

    im_sum = y_true.sum() + y_pred.sum()
    print(im_sum)
    if im_sum == 0:
        return 1.0

    intersection = np.logical_and(y_true, y_pred)
    return 2. * intersection.sum() / im_sum


test_gen = getGenerator(images)

results = model.predict_generator(test_gen, ceil(len(images) / 32), verbose = 1)
dice_scores = []

for i, pred in enumerate(tqdm(results, desc='Saving Masks')):
    m = pred #masks[i]
    p = pred
    s = (dice_coef(m,p))
    dice_scores.append(s)
    # io.imsave(os.path.join(save_path, '%i_pred.png'%i), np.squeeze(pred))

print(np.mean(dice_scores))
