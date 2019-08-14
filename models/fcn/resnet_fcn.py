from losses import *
from keras.optimizers import Adam, SGD
from keras.losses import binary_crossentropy

from keras import layers
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers.core import Activation

from keras.applications.resnet50 import ResNet50
from keras.models import Model

from keras import backend as K
import tensorflow as tf

def getResnet50FCN():
    # load ResNet

    tf.reset_default_graph()
    sess = tf.Session()
    K.clear_session()

    n_classes = 1
    stride = 32

    input_tensor = layers.Input(shape=(256, 256, 1))
    base_model = ResNet50(weights=None, include_top=False, input_tensor=input_tensor)

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
