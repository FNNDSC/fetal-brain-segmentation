from losses import *
from keras.models import Model
from keras.layers import Input, Conv2D, Activation, BatchNormalization, UpSampling2D, MaxPooling2D, Dense, GlobalAveragePooling2D
from keras.optimizers import RMSprop, Adam, SGD
from keras.losses import binary_crossentropy
from keras import backend as K
from keras import layers
import numpy as np

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + (1 - dice_loss(y_true, y_pred))


def down_conv(init, nb_filter):
    x = Conv2D(nb_filter, (3, 3), padding='same', activation='relu', 
        kernel_initializer = 'he_normal')(init)
    x = BatchNormalization()(x)
    x = squeeze_excite_block(x)

    x = MaxPooling2D(pool_size=(2, 2), padding='same')(x)

    return x

def up_conv(init, skip, nb_filter):
    x = UpSampling2D(size = (2,2))(init)
    x = Conv2D(nb_filter, (3, 3), padding='same', activation='relu', 
        kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)
    x = squeeze_excite_block(x)

    x = layers.concatenate([x, skip], axis=3)
    return x

def res_block(init, nb_filter):
    x = Conv2D(nb_filter, (3, 3), padding='same', activation='relu',
        kernel_initializer = 'he_normal')(init)
    x = BatchNormalization()(x)

    x = Conv2D(nb_filter, (3, 3), padding='same', activation='relu',
        kernel_initializer = 'he_normal')(x)
    x = BatchNormalization()(x)

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


def create_model(input_shape):
    inputs = Input(shape=input_shape)
    i = 0

    #0
    x = down_conv(inputs, 32)
    x0 = res_block(x, 32)

    #1
    x = down_conv(x0, 64)
    x1 = res_block(x, 64)

    #2
    x = down_conv(x1, 128)
    x2 = res_block(x, 128)

    #3
    x = down_conv(x2, 256)
    x3 = res_block(x, 256)


    #--------------- center ------------
    x = down_conv(x3, 512)
    x = res_block(x, 512)
    #--------------- center ------------

    #3
    x = up_conv(x, x3, 256)
    x = res_block(x, 256)

    #2
    x = up_conv(x, x2, 128)
    x = res_block(x, 128)

    #1
    x = up_conv(x, x1, 64)
    x = res_block(x, 64)

    #0
    x = up_conv(x, x0, 32)
    x = res_block(x, 32)

    classify = Conv2D(1, (1, 1), activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer = Adam(lr = 1e-4),
                        loss = binary_crossentropy,
                        metrics = [dice_coef])

    return model

def getSEUnet():
    model = create_model((256,256,1))
    return model
