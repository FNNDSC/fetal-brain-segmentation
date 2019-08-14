from losses import *
from keras.optimizers import Adam
from keras.losses import binary_crossentropy

from keras import layers
from keras.models import Model

from keras import backend as K
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


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut"""
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size,
                      padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)

    x = squeeze_excite_block(x)

    return x


def conv_block(input_tensor,
               kernel_size,
               filters,
               stage,
               block,
               strides=(2, 2)):
    """A block that has a conv layer at shortcut"""
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a')(input_tensor)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2c')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)


    shortcut = layers.Conv2D(filters3, (1, 1), strides=strides,
                             kernel_initializer='he_normal',
                             name=conv_name_base + '1')(input_tensor)
    shortcut = layers.BatchNormalization(
        axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = layers.Activation('relu')(x)

    x = squeeze_excite_block(x)

    return x


def getResNet50():
    """Instantiates the ResNet50 architecture.
    """

    bn_axis = 3
    img_input = layers.Input(shape=(256,256,1))

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7),
                      strides=(2, 2),
                      padding='valid',
                      kernel_initializer='he_normal',
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')


    model = Model(img_input, x, name='resnet50')

    return model

def getResnetSE50FCN():

    tf.reset_default_graph()
    sess = tf.Session()
    K.clear_session()

    n_classes = 1
    stride = 32

    base_model = getResNet50()

    # add classifier
    x = base_model.get_layer('activation_49').output
    x = layers.Dropout(0.5)(x)

    x = layers.Conv2D(n_classes,1,name = 'pred_32',padding = 'valid', kernel_initializer='he_normal')(x)

    ## add 32s upsampler

    x = layers.UpSampling2D(size=(stride), interpolation='bilinear')(x)
    x = layers.Activation('sigmoid')(x)
    pred_32s = x

    # 16s
    x = base_model.get_layer('activation_40').output
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(n_classes,1,name = 'pred_16',padding = 'valid', kernel_initializer='he_normal')(x)
    x = layers.UpSampling2D(name='upsampling_16',size=(stride//2), interpolation='bilinear')(x)
    x = layers.Conv2D(n_classes,5,name = 'pred_up_16',padding = 'same', kernel_initializer='he_normal')(x)

    # merge classifiers
    x = layers.add([x, pred_32s])
    x = layers.Activation('sigmoid')(x)
    pred_16s = x

    x = base_model.get_layer('activation_22').output
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


