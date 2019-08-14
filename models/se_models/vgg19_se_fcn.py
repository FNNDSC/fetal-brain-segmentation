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

def getVGG19SE():
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

    model = Model(inputs=img_input,outputs=x)

    return model

def getVGG19SEFCN():

    tf.reset_default_graph()
    sess = tf.Session()
    K.clear_session()

    base_model = getVGG19SE()

    n_classes = 1
    stride = 32
    # add classifier
    x = base_model.get_layer('block5_pool').output
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(n_classes,1,name = 'pred_32',padding = 'valid', kernel_initializer='he_normal')(x)

    ## add 32s upsampler

    x = layers.UpSampling2D(size=(stride), interpolation='bilinear')(x)
    x = layers.Activation('sigmoid')(x)
    pred_32s = x

    # 16s
    x = base_model.get_layer('block4_pool').output
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(n_classes,1,name = 'pred_16',padding = 'valid', kernel_initializer='he_normal')(x)
    x = layers.UpSampling2D(name='upsampling_16',size=(stride//2), interpolation='bilinear')(x)
    x = layers.Conv2D(n_classes,5,name = 'pred_up_16',padding = 'same', kernel_initializer='he_normal')(x)

    # merge classifiers
    x = layers.add([x, pred_32s])
    x = layers.Activation('sigmoid')(x)
    pred_16s = x

    x = base_model.get_layer('block3_pool').output
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(n_classes,1,name = 'pred_8',padding = 'valid', kernel_initializer='he_normal')(x)
    x = layers.UpSampling2D(name='upsampling_8',size=(stride//4), interpolation='bilinear')(x)
    x = layers.Conv2D(n_classes,5,name = 'pred_up_8',padding = 'same', kernel_initializer='he_normal')(x)

    # merge classifiers
    x = layers.add([x, pred_16s])
    x = layers.Activation('sigmoid')(x)

    model = Model(inputs=base_model.input,outputs=x)
    model.compile(optimizer = Adam(lr = 1e-4),
              loss = binary_crossentropy,
              metrics = [dice_coef])

    return model
