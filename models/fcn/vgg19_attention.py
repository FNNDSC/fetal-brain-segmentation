from keras import layers
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras import backend as K
from keras.losses import binary_crossentropy
import tensorflow as tf
from keras.optimizers import Adam

from losses import *

def attention_block(filters, x, shortcut):
    g1 = layers.Conv2D(filters, 1)(x)
    x1 = layers.Conv2D(filters, 1)(shortcut)

    g1_x1 = layers.Add()([g1, x1])
    psi = layers.Activation('relu')(g1_x1)
    psi = layers.Conv2D(1, 1)(psi)
    psi = layers.Activation('sigmoid')(psi)
    x = layers.Multiply()([x1, psi])
    return x

def getVGG19Attention():

    tf.reset_default_graph()
    sess = tf.Session()
    K.clear_session()

    n_classes = 1
    stride = 32

    input_tensor = layers.Input(shape=(256, 256, 1))
    base_model = VGG19(weights=None, include_top=False, input_tensor=input_tensor)

    # add classifier
    x = base_model.get_layer('block5_pool').output
    sh1 = x

    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(n_classes,1,name = 'pred_32',padding = 'valid', kernel_initializer='he_normal')(x)

    ## add 32s upsampler

    x = layers.UpSampling2D(size=(stride), interpolation='bilinear')(x)
    x = layers.Activation('sigmoid')(x)
    pred_32s = x

    # 16s
    x = base_model.get_layer('block4_pool').output
    sh2 = x
    sh1 = layers.UpSampling2D(size=(2), interpolation='bilinear')(sh1)
    x = attention_block(512, sh1, x)
    
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(n_classes,1,name = 'pred_16',padding = 'valid', kernel_initializer='he_normal')(x)
    x = layers.UpSampling2D(name='upsampling_16',size=(stride//2), interpolation='bilinear')(x)
    x = layers.Conv2D(n_classes,5,name = 'pred_up_16',padding = 'same', kernel_initializer='he_normal')(x)

    x = layers.add([x, pred_32s])
    x = layers.Activation('sigmoid')(x)
    pred_16s = x

    # 8s
    x = base_model.get_layer('block3_pool').output
    sh2 = layers.UpSampling2D(size=(2), interpolation='bilinear')(sh2)
    x = attention_block(256, sh2, x)

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

