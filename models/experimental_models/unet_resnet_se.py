from losses import *
from keras.models import Model
from keras.optimizers import RMSprop, Adam, SGD
from keras.losses import binary_crossentropy
from keras import backend as K
from keras import layers
import numpy as np

import tensorflow as tf


def down_conv(init, nb_filter, se_version, no_down = False):
    x = layers.Conv2D(nb_filter, (3, 3), padding='same', activation='relu',
        kernel_initializer = 'he_normal')(init)
    x = layers.BatchNormalization()(x)

    if se_version:
        x = squeeze_excite_block(x)

    if not no_down:
        x = layers.MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    return x

def up_conv(init, skip, nb_filter, se_version):
    x = layers.UpSampling2D(size = (2,2))(init)
    x = layers.Conv2D(nb_filter, (3, 3), padding='same', activation='relu',
        kernel_initializer = 'he_normal')(x)
    x = layers.BatchNormalization()(x)

    if se_version:
        x = squeeze_excite_block(x)

    x = layers.concatenate([x, skip], axis=3)
    return x

def res_block(init, nb_filter, se_version):
    x = layers.Conv2D(nb_filter, (3, 3), padding='same', activation='relu',
        kernel_initializer = 'he_normal')(init)
    x = layers.BatchNormalization()(x)

    x = layers.Conv2D(nb_filter, (3, 3), padding='same', activation='relu',
        kernel_initializer = 'he_normal')(x)
    x = layers.BatchNormalization()(x)

    if se_version:
        x = squeeze_excite_block(x)

    x = layers.concatenate([init, x], axis=3)
    return x

def squeeze_excite_block(input, ratio=16):
    init = input
    channel_axis = -1
    filters = init._keras_shape[channel_axis]
    se_shape = (1, 1, filters)

    se = layers.GlobalAveragePooling2D()(init)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation='relu')(se)
    se = layers.Dense(filters, activation='sigmoid')(se)

    x = layers.multiply([init, se])
    return x


def create_model(input_shape, se_version):
    inputs = layers.Input(shape=input_shape)
    i = 0

    #0
    x = down_conv(inputs, 32, se_version)
    x0 = res_block(x, 32, se_version)

    #1
    x = down_conv(x0, 64, se_version)
    x1 = res_block(x, 64, se_version)

    #2
    x = down_conv(x1, 128, se_version)
    x2 = res_block(x, 128, se_version)

    #3
    x = down_conv(x2, 256, se_version)
    x3 = res_block(x, 256, se_version)
    x3 = layers.Dropout(0.5)(x3)

    #--------------- center ------------
    x = down_conv(x3, 512, se_version)
    x = res_block(x, 512, se_version)
    x = layers.Dropout(0.5)(x)
    #--------------- center ------------

    #3
    x = up_conv(x, x3, 256, se_version)
    x = res_block(x, 256, se_version)

    #2
    x = up_conv(x, x2, 128, se_version)
    x = res_block(x, 128, se_version)

    #1
    x = up_conv(x, x1, 64, se_version)
    x = res_block(x, 64, se_version)

    #0
    x = up_conv(x, x0, 32, se_version)
    x = res_block(x, 32, se_version)

    x = up_conv(x, inputs, 16, se_version)
    x = res_block(x, 16, se_version)

    classify = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer = Adam(lr = 1e-4),
                        loss = binary_crossentropy,
                        metrics = [dice_coef])

    return model

def getUnetRes(se_version=False):

    tf.reset_default_graph()
    sess = tf.Session()
    K.clear_session()

    model = create_model((256,256,1), se_version)
    #print(model.summary())
    return model
