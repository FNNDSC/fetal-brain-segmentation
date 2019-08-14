import numpy as np
from keras.preprocessing.image import ImageDataGenerator

def resetSeed():
    np.random.seed(1)


#gets image generators for training and validation
def getGenerator(images, masks, augmentation = False, batch_size=32):
    resetSeed()
    seed = 1

    #if there is no augmentation the img is only rescaled by
    #dividing by 255 this ensures a range of 0-1 for all pixel values
    if augmentation:
        data_gen_args = dict(rescale=1./255,
            horizontal_flip = True,
            vertical_flip = True,
            rotation_range = 90,
            brightness_range = (0.5, 1.5))
    else:
        data_gen_args = dict(rescale=1./255)

    #do the same for img and masks to ensure mask stays the same
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    image_datagen.fit(images, augment=True, seed=seed)
    mask_datagen.fit(masks, augment=True, seed=seed)

    save_dir = './data/augmented/'

    image_generator = image_datagen.flow(x = images, batch_size=batch_size, seed=seed)
            # save_to_dir=save_dir)
    mask_generator = mask_datagen.flow(x = masks, batch_size=batch_size, seed=seed)
            # save_to_dir=save_dir)

    generator = zip(image_generator, mask_generator)

    return generator

