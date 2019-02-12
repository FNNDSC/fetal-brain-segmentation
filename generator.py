import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def resetSeed():
    np.random.seed(1)

def getGenerator(images, masks, augmentation = False):
    resetSeed()
    seed = 1

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

