import numpy as np

import os
import skimage.io as io
from tqdm import tqdm
from datahandler import DataHandler
from unet import *
from unetVGG19 import *
from keras.models import *
from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator

model_name = 'unetVGG19'
has_3_channels = False

if model_name is 'unetVGG19':
    has_3_channels = True

def resetSeed():
    np.random.seed(1)

def getGenerator(images, masks, augmentation = False):
    resetSeed()
    seed = 1

    # TODO add augmentation

    if augmentation:
        data_gen_args = dict(rescale=1./255,
            rotation_range=60,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.2)
    else:
        data_gen_args = dict(rescale=1./255)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_datagen.fit(images, augment=True, seed=seed)
    mask_datagen.fit(masks, augment=True, seed=seed)

    save_dir = './data/augmented/'

    image_generator = image_datagen.flow(x = images, batch_size=16, seed=seed)
    mask_generator = mask_datagen.flow(x = masks, batch_size=16, seed=seed)

    generator = zip(image_generator, mask_generator)

    return generator

resetSeed()

epochs = 50
batch_size = 16

dh = DataHandler()
tr_images, tr_masks, te_images, te_masks = dh.getData(has_3_channels = has_3_channels)


# TODO remove this
# save_path = './data/as_read/'
# for i, img in enumerate(tqdm(tr_images, desc='Saving Imgs')):
#     io.imsave(os.path.join(save_path,"%d_img.png"%i), np.squeeze(img))
#     io.imsave(os.path.join(save_path,"%d_msk.png"%i), np.squeeze(tr_masks[i]))

train_generator = getGenerator(tr_images, tr_masks, augmentation = True)
val_generator = getGenerator(te_images, te_masks, augmentation = False)

if model_name is 'unetVGG19':
    cp_name = 'unetVGG19_brain_seg.h5'
    log_name = 'log_unetVGG19.csv'
    model = getUnetVGG19()

else:
    cp_name = 'unet_brain_seg.h5'
    log_name = 'log_unet.csv'
    #my unet
    model = getUnet()
    #load weights from other problem transfer learning
    model.load_weights('unet_transfer.h5')

print(model.summary())

model_json = model.to_json()

with open("model.json", "w") as json_file:
     json_file.write(model_json)

checkpoint = ModelCheckpoint(cp_name,
        monitor='val_loss',
        verbose=1,
        save_best_only=True,
        save_weights_only=False,
        period=1)

early = EarlyStopping(monitor='val_loss',
        min_delta=0,
        patience=10,
        verbose=1,
        restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
        factor=0.1,
        patience=3,
        min_lr=0.000001,
        verbose=1)

csv_logger = CSVLogger(log_name, separator=',', append=True)

history = model.fit_generator(train_generator,
        epochs=epochs,
        steps_per_epoch = len(tr_images) / batch_size,
        validation_data = val_generator,
        validation_steps = len(te_images) / batch_size,
        verbose = 1,
        callbacks = [checkpoint, early, reduce_lr, csv_logger])

