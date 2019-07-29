import numpy as np
from losses import *
import keras
from keras.models import *
from keras import layers
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

def getUnetUpconv():

    tf.reset_default_graph()
    sess = tf.Session()
    K.clear_session()

    inputs = layers.Input((256, 256, 1))

    # Encoding (downwards)
    conv1 = layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l2(0.0005))(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l2(0.0005))(conv1)
    conv1 = layers.BatchNormalization()(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l2(0.0005))(pool1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l2(0.0005))(conv2)
    conv2 = layers.BatchNormalization()(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l2(0.0005))(pool2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l2(0.0005))(conv3)
    conv3 = layers.BatchNormalization()(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l2(0.0005))(pool3)
    conv4 = layers.BatchNormalization()(conv4)
    conv4 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l2(0.0005))(conv4)
    conv4 = layers.BatchNormalization()(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    #flat
    conv5 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l2(0.0005))(pool4)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l2(0.0005))(conv5)
    conv5 = layers.BatchNormalization()(conv5)

    # Decoding (upwards)
    up6 = layers.Conv2DTranspose(256, 2, strides=(2,2), activation='relu', kernel_initializer='he_normal', activity_regularizer=regularizers.l2(0.0005))(conv5)
    up6 = layers.BatchNormalization()(up6)
    merge6 = layers.concatenate([conv4,up6], axis = 3)
    conv6 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l2(0.0005))(merge6)
    conv6 = layers.BatchNormalization()(conv6)
    conv6 = layers.Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l2(0.0005))(conv6)
    conv6 = layers.BatchNormalization()(conv6)

    up7 = layers.Conv2DTranspose(128, 2, strides=(2,2), activation='relu', kernel_initializer='he_normal', activity_regularizer=regularizers.l2(0.0005))(conv6)
    up7 = layers.BatchNormalization()(up7)
    merge7 = layers.concatenate([conv3,up7], axis = 3)
    conv7 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l2(0.0005))(merge7)
    conv7 = layers.BatchNormalization()(conv7)
    conv7 = layers.Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l2(0.0005))(conv7)
    conv7 = layers.BatchNormalization()(conv7)

    up8 = layers.Conv2DTranspose(64, 2, strides=(2,2), activation='relu', kernel_initializer='he_normal', activity_regularizer=regularizers.l2(0.0005))(conv7)
    up8 = layers.BatchNormalization()(up8)
    merge8 = layers.concatenate([conv2,up8], axis = 3)
    conv8 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l2(0.0005))(merge8)
    conv8 = layers.BatchNormalization()(conv8)
    conv8 = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l2(0.0005))(conv8)
    conv8 = layers.BatchNormalization()(conv8)

    up9 = layers.Conv2DTranspose(32, 2, strides=(2,2), activation='relu', kernel_initializer='he_normal', activity_regularizer=regularizers.l2(0.0005))(conv8)
    up9 = layers.BatchNormalization()(up9)
    merge9 = layers.concatenate([conv1,up9], axis = 3)
    conv9 = layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l2(0.0005))(merge9)
    conv9 = layers.BatchNormalization()(conv9)
    conv9 = layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', activity_regularizer=regularizers.l2(0.0005))(conv9)
    conv9 = layers.BatchNormalization()(conv9)

    conv10 = layers.Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer = Adam(lr = 1e-4),
        loss=binary_crossentropy, #[binary_focal_loss(gamma=1., alpha=0.6)],
        metrics = [dice_coef])

    return model
