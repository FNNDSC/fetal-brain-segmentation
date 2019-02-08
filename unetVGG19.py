import numpy as np
import keras
from keras.models import *
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras.applications import vgg19
import tensorflow as tf

def dice_coef(y_true, y_pred, smooth=1):
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)
    intersection = K.sum(flat_y_true * flat_y_pred)
    return (2. * intersection + smooth) / (K.sum(flat_y_true) + K.sum(flat_y_pred) + smooth)

def dice_border_loss(y_true, y_pred):
    return (1 - ((border_dice_coef(y_true, y_pred) * 0.05)
        + (dice_coef(y_true, y_pred) * 0.95)))

# calculates the dice coef of the area around the border
def border_dice_coef(y_true, y_pred):
    border = get_border(y_true)
    flat_border = K.flatten(border)
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)

    # only get values around border
    border_y_true = K.tf.gather(flat_y_true, K.tf.where(border > 0.5))
    border_y_pred = K.tf.gather(flat_y_pred, K.tf.where(border > 0.5))

    return dice_coef(border_y_true, border_y_pred)

#returns a mask of the area around the masks border
def get_border(y_true):
    mask = (25,25)
    pos = y_true
    neg = 1-y_true
    #padding same mantains same size
    #pool_size big to keep all area around mask
    pos = K.pool2d(pos, pool_size=mask, padding='same')
    neg = K.pool2d(neg, pool_size=mask, padding='same')

    border = pos * neg
    return border

def bce_dice_loss(y_true, y_pred):
    return 0.5 * keras.losses.binary_crossentropy(y_true, y_pred) - dice_coef(y_true, y_pred)

def getUnetVGG19():

    tf.reset_default_graph()
    sess = tf.Session()

    # encoder vgg19
    vgg19_model = vgg19.VGG19(include_top = False, weights = None, #'imagenet',
            input_shape = (256, 256, 3), classes = 2)


    # block 6
    x = Conv2D(1024, 3,
            activation = 'relu',
            padding = 'same',
            name = 'block6_conv1')(vgg19_model.output)

    x = Conv2D(1024, 3,
            activation = 'relu',
            padding = 'same',
            name = 'block6_conv2')(x)

    x = Dropout(0.5, name = 'block6_dropout')(x)

    # decoder block 7
    x = Conv2D(512, 2, activation = 'relu', padding = 'same')(
            UpSampling2D(size = (2,2))(x))
    x = concatenate([vgg19_model.get_layer('block5_conv4').output,x], axis = 3)
    x = Conv2D(512, 3, activation = 'relu', padding = 'same', name = 'block7_conv1')(x)
    x = Conv2D(512, 3, activation = 'relu', padding = 'same', name = 'block7_conv2')(x)
    x = Conv2D(512, 3, activation = 'relu', padding = 'same', name = 'block7_conv3')(x)
    x = Conv2D(512, 3, activation = 'relu', padding = 'same', name = 'block7_conv4')(x)

    # decoder block 8
    x = Conv2D(512, 2, activation = 'relu', padding = 'same')(
            UpSampling2D(size = (2,2))(x))
    x = concatenate([vgg19_model.get_layer('block4_conv4').output,x], axis = 3)
    x = Conv2D(512, 3, activation = 'relu', padding = 'same', name = 'block8_conv1')(x)
    x = Conv2D(512, 3, activation = 'relu', padding = 'same', name = 'block8_conv2')(x)
    x = Conv2D(512, 3, activation = 'relu', padding = 'same', name = 'block8_conv3')(x)
    x = Conv2D(512, 3, activation = 'relu', padding = 'same', name = 'block8_conv4')(x)

    # decoder block 9
    x = Conv2D(256, 2, activation = 'relu', padding = 'same')(
            UpSampling2D(size = (2,2))(x))
    x = concatenate([vgg19_model.get_layer('block3_conv4').output,x], axis = 3)
    x = Conv2D(256, 3, activation = 'relu', padding = 'same', name = 'block9_conv1')(x)
    x = Conv2D(256, 3, activation = 'relu', padding = 'same', name = 'block9_conv2')(x)
    x = Conv2D(256, 3, activation = 'relu', padding = 'same', name = 'block9_conv3')(x)
    x = Conv2D(256, 3, activation = 'relu', padding = 'same', name = 'block9_conv4')(x)

    # decoder block 10
    x = Conv2D(128, 2, activation = 'relu', padding = 'same')(
            UpSampling2D(size = (2,2))(x))
    x = concatenate([vgg19_model.get_layer('block2_conv2').output,x], axis = 3)
    x = Conv2D(128, 3, activation = 'relu', padding = 'same', name = 'block10_conv1')(x)
    x = Conv2D(128, 3, activation = 'relu', padding = 'same', name = 'block10_conv2')(x)

    # decoder block 11
    x = Conv2D(64, 2, activation = 'relu', padding = 'same')(
            UpSampling2D(size = (2,2))(x))
    x = concatenate([vgg19_model.get_layer('block1_conv2').output,x], axis = 3)
    x = Conv2D(64, 3, activation = 'relu', padding = 'same', name = 'block11_conv1')(x)
    x = Conv2D(64, 3, activation = 'relu', padding = 'same', name = 'block11_conv2')(x)

    x = Conv2D(2, 3, activation = 'relu', padding = 'same')(x)
    x = Conv2D(1, 1, activation = 'sigmoid')(x)

    model = Model(vgg19_model.input, x, name = 'UnetVGG19')
    model.compile(optimizer = Adam(lr = 1e-3),
        loss = bce_dice_loss,
        metrics = [dice_coef])

    return model

