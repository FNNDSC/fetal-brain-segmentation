import numpy as np
from losses import *
import keras
from keras.models import *
from keras import layers

from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

from keras.losses import binary_crossentropy

import tensorflow as tf


def squeeze_excite_block(input, ratio=16):
    init = input
    channel_axis = -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = layers.GlobalAveragePooling2D()(init)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = layers.multiply([init, se])
    return x


def getSEUnetUpconv():

    tf.reset_default_graph()
    sess = tf.Session()
    K.clear_session()

    inputs = layers.Input((256, 256, 1))

    # Encoding (downwards)
    conv1 = layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    se1 = squeeze_excite_block(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(se1)

    conv2 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    se2 = squeeze_excite_block(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(se2)

    conv3 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    se3 = squeeze_excite_block(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(se3)

    conv4 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    se4 = squeeze_excite_block(conv4)
    drop4 = layers.Dropout(0.5)(se4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(drop4)

    #flat
    conv5 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    se5 = squeeze_excite_block(conv5)
    drop5 = layers.Dropout(0.5)(se5)

    # Decoding (upwards)
    up6 = layers.Conv2DTranspose(256, 2, strides=(2,2), activation='relu', kernel_initializer='he_normal')(se5)
    merge6 = layers.concatenate([conv4,up6], axis = 3)
    conv6 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    se6 = squeeze_excite_block(conv6)

    up7 = layers.Conv2DTranspose(128, 2, strides=(2,2), activation='relu', kernel_initializer='he_normal')(se6)
    merge7 = layers.concatenate([conv3,up7], axis = 3)
    conv7 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
    se7 = squeeze_excite_block(conv7)

    up8 = layers.Conv2DTranspose(64, 2, strides=(2,2), activation='relu', kernel_initializer='he_normal')(se7)
    merge8 = layers.concatenate([conv2,up8], axis = 3)
    conv8 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    se8 = squeeze_excite_block(conv8)

    up9 = layers.Conv2DTranspose(32, 2, strides=(2,2), activation='relu', kernel_initializer='he_normal')(se8)
    merge9 = layers.concatenate([conv1,up9], axis = 3)
    conv9 = layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)

    conv10 = layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = 1e-4),
        loss=binary_crossentropy, #[binary_focal_loss(gamma=1., alpha=0.6)],
        metrics = [dice_coef])

    return model
