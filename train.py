import numpy as np

import os
import skimage.io as io
from tqdm import tqdm
from datahandler import DataHandler
from unet import *

from keras.models import *
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import warnings

warnings.filterwarnings("ignore")

def resetSeed():
    np.random.seed(1)

def getGenerator(images, masks):
    resetSeed()
    seed = 1

    # TODO add augmentation
    data_gen_args = dict(rescale=1./255)
   # ,
    #         featurewise_center=True,
    #         featurewise_std_normalization=True,
    #         rotation_range=90,
    #         width_shift_range=0.1,
    #         height_shift_range=0.1,
    #         zoom_range=0.2)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_datagen.fit(images, augment=True, seed=seed)
    mask_datagen.fit(masks, augment=True, seed=seed)

    save_dir = './data/augmented/'

    image_generator = image_datagen.flow(x = images, seed=seed,
            shuffle=False) #, save_to_dir = save_dir, save_prefix = 'img')
    mask_generator = mask_datagen.flow(x = masks, seed=seed,
            shuffle=False) #, save_to_dir = save_dir, save_prefix = 'mask')

    generator = zip(image_generator, mask_generator)

    return generator

resetSeed()

epochs = 50
batch_size = 32

dh = DataHandler()
tr_images, tr_masks, te_images, te_masks = dh.getData()


# TODO remove this
save_path = './data/as_read/'
for i, img in enumerate(tqdm(tr_images, desc='Saving Imgs')):
    io.imsave(os.path.join(save_path,"%d_img.png"%i), np.squeeze(img))
    io.imsave(os.path.join(save_path,"%d_msk.png"%i), np.squeeze(tr_masks[i]))

train_generator = getGenerator(tr_images, tr_masks)
val_generator = getGenerator(te_images, te_masks)

model = getUnet()

#load weights from other problem transfer learning
model.load_weights('unet_transfer.h5')

model_json = model.to_json()

with open("model.json", "w") as json_file:
     json_file.write(model_json)

check_point_name = "unet_brain_seg.h5"

checkpoint = ModelCheckpoint(check_point_name,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        mode='min',
        period=1)

early = EarlyStopping(monitor='val_loss',
        min_delta=0.001,
        patience=10,
        verbose=1,
        mode='min')

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
        factor=0.1,
        patience=3,
        min_lr=0.000001,
        verbose=1)

history = model.fit_generator(train_generator,
        epochs=epochs,
        steps_per_epoch = len(tr_images) / batch_size,
        validation_data = val_generator,
        validation_steps = len(te_images) / batch_size,
        verbose = 1,
        callbacks = [checkpoint, early, reduce_lr])

