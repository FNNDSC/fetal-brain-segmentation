import numpy as np
import keras
from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from keras.layers import SpatialDropout2D, Dense, Reshape, GlobalAveragePooling2D, Multiply, Activation, Add
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import regularizers

from losses import *

from keras.losses import binary_crossentropy

import tensorflow as tf

def attention_block(filters, x, shortcut):
    g1 = Conv2D(filters, 1)(x)
    x1 = Conv2D(filters, 1)(shortcut)

    g1_x1 = Add()([g1, x1])
    psi = Activation('relu')(g1_x1)
    psi = Conv2D(1, 1)(psi)
    psi = Activation('sigmoid')(psi)
    x = Multiply()([x1, psi])
    return x

def getAttentionUnetBN(loss_function = 'dice'):

    tf.reset_default_graph()
    sess = tf.Session()
    K.clear_session()

    inputs = Input((256, 256, 1))

    # Encoding (downwards)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same',
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(inputs)
    conv1 = Conv2D(32, 3, activation = 'relu', padding = 'same',
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(conv1)
    conv1 = SpatialDropout2D(0.1)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same',
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(pool1)
    conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same',
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same',
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(pool2)
    conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same',
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same',
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(pool3)
    conv4 = Conv2D(256, 3, activation = 'relu', padding = 'same',
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    #flat
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same',
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(pool4)
    conv5 = Conv2D(512, 3, activation = 'relu', padding = 'same',
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(conv5)
    drop5 = Dropout(0.5)(conv5)

    # Decoding (upwards)
    up1 = UpSampling2D(size = (2,2))(drop5)
    att1 = attention_block(256, up1, drop4)
    merge6 = concatenate([att1, up1], axis = 3)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same',
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(merge6)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same',
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(conv6)

    up2 = UpSampling2D(size = (2,2))(conv6)
    att2 = attention_block(128, up2, conv3)
    merge7 = concatenate([att2, up2], axis = 3)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same',
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(merge7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same',
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(conv7)

    up3 = UpSampling2D(size = (2,2))(conv7)
    att3 = attention_block(64, up3, conv2)
    merge8 = concatenate([att3, up3], axis = 3)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same',
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same',
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(conv8)

    up4 = UpSampling2D(size=(2,2))(conv8)
    att4 = attention_block(32, up4, conv1)
    merge9 = concatenate([att4, up4], axis = 3)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same',
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(merge9)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same',
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(conv9)

    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    if loss_function == 'dice':
        loss = dice_loss
    elif loss_function == 'BCE_DICE':
        loss = bce_dice_loss

    model.compile(optimizer = Adam(lr = 1e-4),
                        loss = loss,
                        metrics = [dice_coef])

    return model


