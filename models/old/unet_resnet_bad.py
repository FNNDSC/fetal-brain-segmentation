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

def identity_block_same(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut"""
    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters1, (1, 1),
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2a', padding='same')(input_tensor)
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
                      name=conv_name_base + '2c', padding='same')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = layers.Activation('relu')(x)
    return x

def identity_block(input_tensor, kernel_size, filters, stage, block, se_version = False):
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
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2), se_version = False):
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

    if se_version:
        x = squeeze_excite_block(x)

    return x

def up_conv_block(input_tensor, skip_target, kernel_size, filters,
    stage, block, strides=(2, 2), se_version = False):

    filters1, filters2, filters3 = filters
    bn_axis = 3

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = layers.UpSampling2D(size = (2,2))(input_tensor)

    x = layers.Conv2D(filters1, (1, 1), strides=strides,
                      kernel_initializer='he_normal',
                      padding='same',
                      name=conv_name_base + '2a_up')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2a_up')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters2, kernel_size, padding='same',
                      kernel_initializer='he_normal',
                      name=conv_name_base + '2b_up')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2b_up')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(filters3, (1, 1),
                      kernel_initializer='he_normal',
                      padding='same',
                      name=conv_name_base + '2c_up')(x)
    x = layers.BatchNormalization(axis=bn_axis, name=bn_name_base + '2c_up')(x)

    shortcut = layers.Conv2D(filters3//2, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layers.UpSampling2D(size = (2,2))(input_tensor))
    shortcut = layers.concatenate([skip_target,shortcut], axis = 3)

    x = layers.add([x, shortcut])
    x = layers.Conv2D(filters3//2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(x)

    if se_version:
        x = squeeze_excite_block(x)

    return x

def UnetResNet18(se_version):
    """Instantiates the ResNet18 architecture.
    """

    tf.reset_default_graph()
    sess = tf.Session()
    K.clear_session()

    bn_axis = 3
    img_input = layers.Input(shape=(256,256,1))

    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal', name='conv1')(x)

    x = layers.BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = layers.Activation('relu')(x)
    x1 = x

    x = layers.ZeroPadding2D(padding=(1, 1), name='pool1_pad')(x)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    if se_version:
        x = squeeze_excite_block(x)

    x = conv_block(x, 3, [64, 64, 64], stage=2, block='a', strides=(1, 1), se_version=se_version)
    x = identity_block(x, 3, [64, 64, 64], stage=2, block='b', se_version=se_version)
    x2 = x

    x = conv_block(x, 3, [128, 128, 128], stage=3, block='a', se_version=se_version)
    x = identity_block(x, 3, [128, 128, 128], stage=3, block='b', se_version=se_version)
    x = identity_block(x, 3, [128, 128, 128], stage=3, block='c', se_version=se_version)
    x = identity_block(x, 3, [128, 128, 128], stage=3, block='d', se_version=se_version)
    x3 = x

    x = conv_block(x, 3, [256, 256, 256], stage=4, block='a', se_version=se_version)
    x = identity_block(x, 3, [256, 256, 256], stage=4, block='b', se_version=se_version)
    x4 = x

    x = conv_block(x, 3, [512, 512, 512], stage=5, block='a', se_version=se_version)
    x = identity_block(x, 3, [512, 512, 512], stage=5, block='b', se_version=se_version)

    # Decoding (upwards)
    x = up_conv_block(x, x4, 3, [256, 256, 512], stage=6, block='a_up', strides=(1,1), se_version=se_version)
    x = identity_block(x, 3, [256, 256, 256], stage=6, block='b_up', se_version=se_version)

    x = up_conv_block(x, x3, 3, [128, 128, 256], stage=7, block='a_up', strides=(1,1), se_version=se_version)
    x = identity_block(x, 3, [128, 128, 128], stage=7, block='b_up', se_version=se_version)
    x = identity_block(x, 3, [128, 128, 128], stage=7, block='c_up', se_version=se_version)
    x = identity_block(x, 3, [128, 128, 128], stage=7, block='d_up', se_version=se_version)

    x = up_conv_block(x, x2, 3, [64, 64, 128], stage=8, block='a_up', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 64], stage=8, block='b')

    x = layers.Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layers.UpSampling2D(size = (2,2))(x))
    x = layers.concatenate([x,x1], axis = 3)
    x = layers.Conv2D(32,3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    x = layers.Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(layers.UpSampling2D(size = (2,2))(x))

    x = layers.Conv2D(1,3, activation='relu', padding='same', kernel_initializer='he_normal')(x)

    model = Model(img_input, x, name='resnet18')

    return model

def getUnetResnet18(se_version=False):
    model = UnetResNet18(se_version)

    model.compile(optimizer = Adam(lr = 1e-4),
            loss = binary_crossentropy,
            metrics = [dice_coef])

    return model

