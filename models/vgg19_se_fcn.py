from keras import layers
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras import backend as K
from keras.losses import binary_crossentropy
import tensorflow as tf
from keras.optimizers import Adam

from losses import *

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

def VGG19SE():
    img_input = layers.Input(shape=(256,256,1))

    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1')(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)
    x = squeeze_excite_block(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1')(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)
    x = squeeze_excite_block(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3')(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)
    x = squeeze_excite_block(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)
    x = squeeze_excite_block(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3')(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv4')(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    x = squeeze_excite_block(x)

    model = Model(input=img_input,output=x)

    return model

def getVGG19SEFCN():
    base_model = VGG19SE()

    n_classes = 1
    stride = 32

    input_tensor = layers.Input(shape=(256, 256, 1))
    base_model = VGG19(weights=None, include_top=False, input_tensor=input_tensor)

    #32
    pool_5 = base_model.get_layer('block5_pool').output
    up_32 = layers.Conv2DTranspose(n_classes, 3, name='up_32', strides=(stride), activation='relu')(pool_5)
    bn1 = layers.BatchNormalization(name='BN_1')(up_32)
    pred_32 = layers.Conv2D(n_classes, 3, name='pred_32', padding = 'same', activation='sigmoid')(bn1)

    #16
    pool_4 = base_model.get_layer('block4_pool').output
    up_16 = layers.Conv2DTranspose(n_classes, 3, name='up_16', strides=(stride//2), activation='relu')(pool_4)
    bn2 = layers.BatchNormalization(name='BN_2')(up_16)
    addition_1 = layers.add([bn2, pred_32])
    pred_16 = layers.Conv2D(n_classes, 3, name='pred_16', padding = 'same', activation='sigmoid')(addition_1)

    #8
    pool_3 = base_model.get_layer('block3_pool').output
    up_8 = layers.Conv2DTranspose(n_classes, 3, name='up_8', strides=(stride//4), activation='relu')(pool_3)
    bn3 = layers.BatchNormalization(name='BN_3')(up_8)
    addition_2 = layers.add([bn3, pred_16])
    pred_8 = layers.Conv2D(n_classes, 3, name='pred_8', padding = 'same', activation='sigmoid')(addition_2)

    x = pred_8

    model = Model(input=base_model.input,output=x)
    model.compile(optimizer = Adam(lr = 1e-4),
              loss = binary_crossentropy,
              metrics = [dice_coef])

    return model
