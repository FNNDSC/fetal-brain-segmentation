import numpy as np
from losses import *
import keras
from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate, SpatialDropout2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras import regularizers

from keras.losses import binary_crossentropy

import tensorflow as tf

def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


    return binary_focal_loss_fixed

def getUnetBN(loss_function = 'BCE'):

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
    up6 = Conv2D(256, 2, activation = 'relu', padding = 'same', 
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', 
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(merge6)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', 
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(conv6)

    up7 = Conv2D(128, 2, activation = 'relu', padding = 'same', 
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(merge7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', 
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(conv7)

    up8 = Conv2D(64, 2, activation = 'relu', padding = 'same', 
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', 
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', 
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(conv8)

    up9 = Conv2D(32, 2, activation = 'relu', padding = 'same', 
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', 
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(merge9)
    conv9 = Conv2D(32, 3, activation = 'relu', padding = 'same', 
        kernel_initializer = 'he_normal', kernel_regularizer=regularizers.l2(0.0001))(conv9)

    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    if loss_function == 'BCE':
        loss = binary_crossentropy
    elif loss_function == 'focal'
        loss = binary_focal_loss(gamma=1., alpha=0.6)
    elif loss_function == 'dice':
        loss = dice_loss
    elif loss_function == 'BCE_DICE':
        loss = bce_dice_loss

    print(loss_function)

    model.compile(optimizer = Adam(lr = 1e-4),
                        loss=loss,
                        metrics = [dice_coef])

    return model
