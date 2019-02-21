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
model.load_weights('logs/unet/unetBceDice/unetBceDice_weights.h5')

dh = DataHandler()
images, masks = dh.getData(only_test = True)

save_path = './data/test_results/'
for i, img in enumerate(tqdm(images, desc='Saving Imgs')):
    io.imsave(os.path.join(save_path,"%d_img.png"%i), np.squeeze(img))
    io.imsave(os.path.join(save_path,"%d_mask.png"%i), np.squeeze(masks[i]))

def resetSeed():
    np.random.seed(1)

def getGenerator(images):
    resetSeed()

    image_datagen = ImageDataGenerator(rescale=1./255)
    image_datagen.fit(images, augment = True)
    image_generator = image_datagen.flow(x = images,
            shuffle = False)

    return image_generator

test_gen = getGenerator(images)

results = model.predict_generator(test_gen, ceil(len(images) / 16), verbose = 1)

for i, mask in enumerate(tqdm(results, desc='Saving Masks')):
    print(mask.shape)
    print(np.min(mask))
    print(np.max(mask))
    io.imsave(os.path.join(save_path, '%i_prediction.png'%i), np.squeeze(mask))
