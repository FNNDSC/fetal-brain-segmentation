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

    input_tensor = Input(shape=(256, 256, 1))
    base_model = ResNet50(weights=None, include_top=False, input_tensor=input_tensor)

    #32
    act_49 = base_model.get_layer('activation_49').output
    up_32 = layers.Conv2DTranspose(n_classes, 3, name='up_32', strides=(stride), activation='relu', kernel_initializer = 'he_normal')(act_49)
    bn1 = layers.BatchNormalization(name='BN_1')(up_32)
    pred_32 = layers.Conv2D(n_classes, 3, name='pred_32', padding = 'same', activation='sigmoid', kernel_initializer = 'he_normal')(bn1)

    #16
    act_40 = base_model.get_layer('activation_40').output
    up_16 = layers.Conv2DTranspose(n_classes, 3, name='up_16', strides=(stride//2), activation='relu', kernel_initializer = 'he_normal')(act_40)
    bn2 = layers.BatchNormalization(name='BN_2')(up_16)
    addition_1 = layers.add([bn2, pred_32])
    pred_16 = layers.Conv2D(n_classes, 3, name='pred_16', padding = 'same', activation='sigmoid', kernel_initializer = 'he_normal')(addition_1)

    #8
    act_22 = base_model.get_layer('activation_22').output
    up_8 = layers.Conv2DTranspose(n_classes, 3, name='up_8', strides=(stride//4), activation='relu', kernel_initializer = 'he_normal')(act_22)
    bn3 = layers.BatchNormalization(name='BN_3')(up_8)
    addition_2 = layers.add([bn3, pred_16])
    pred_8 = layers.Conv2D(n_classes, 3, name='pred_8', padding = 'same', activation='sigmoid', kernel_initializer = 'he_normal')(addition_2)

    x = pred_8

    model = Model(input=base_model.input,output=x)

    model.compile(optimizer = Adam(lr = 1e-4),
                loss = binary_crossentropy,
                metrics = [dice_coef])

    return model
