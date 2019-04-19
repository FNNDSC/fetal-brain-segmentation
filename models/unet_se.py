from losses import *
from keras.models import Model
from keras.layers import Input, Conv2D, Activation, BatchNormalization, Conv2DTranspose, MaxPooling2D, Dense, GlobalAveragePooling2D
from keras.optimizers import RMSprop, Adam, SGD
from keras.losses import binary_crossentropy
from keras import backend as K
from keras import layers
import numpy as np

def dice_loss(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    return binary_crossentropy(y_true, y_pred) + (1 - dice_loss(y_true, y_pred))


def dw_conv(init, nb_filter):
    residual = Conv2D(nb_filter, (1, 1), strides=(2, 2), padding='same', activation='relu')(init)
    residual = BN(residual)

    x = Conv2D(nb_filter, (3, 3), padding='same', activation='relu')(init)
    x = BN(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter, (3, 3), padding='same', activation='relu')(x)
    x = BN(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = layers.add([x, residual])

    return x


def up_conv(init, skip, nb_filter):
    x = Conv2DTranspose(nb_filter, (3, 3), padding='same', strides=(2, 2))(init)
    x = BN(x)
    x = layers.add([x, skip])
    return x


def BN(x):
    return BatchNormalization()(x)


def res_block(init, nb_filter):
    x = Activation('relu')(init)

    x = Conv2D(nb_filter, (3, 3), padding='same', activation='relu')(x)
    x = BN(x)
    x = Activation('relu')(x)

    x = Conv2D(nb_filter, (3, 3), padding='same', activation='relu')(x)
    x = BN(x)

    x = Squeeze_excitation_layer(x)

    x = layers.add([init, x])
    return x


def Squeeze_excitation_layer(input_x):
    ratio = 16
    out_dim =  int(np.shape(input_x)[-1])
    squeeze = GlobalAveragePooling2D()(input_x)
    excitation = Dense(units=int(out_dim / ratio))(squeeze)
    excitation = Activation('relu')(excitation)
    excitation = Dense(units=out_dim)(excitation)
    excitation = Activation('sigmoid')(excitation)
    excitation = layers.Reshape([-1,1,out_dim])(excitation)
    scale = layers.multiply([input_x, excitation])

    return scale


def create_model(input_shape):
    inputs = Input(shape=input_shape)
    i = 0

    nb_filter = [32, 64, 128, 256, 512, 256, 128, 64, 32]

    #0
    x = Conv2D(nb_filter[i], (3, 3), padding='same', activation='relu')(inputs)
    x = BN(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter[i], (3, 3), padding='same', activation='relu')(x)
    x0 = BN(x)
    x = Activation('relu')(x0)
    i += 1

    #1
    x = dw_conv(x0, nb_filter[i])
    # x = res_block(x, nb_filter[i])
    x1 = res_block(x, nb_filter[i])
    i += 1

    #2
    x = dw_conv(x1, nb_filter[i])
    # x = res_block(x, nb_filter[i])
    x2 = res_block(x, nb_filter[i])
    i += 1

    #3
    x = dw_conv(x2, nb_filter[i])
    # x = res_block(x, nb_filter[i])
    x3 = res_block(x, nb_filter[i])
    i += 1


    #--------------- center ------------
    x = dw_conv(x3, nb_filter[i])
    # x = res_block(x, nb_filter[i])
    x = res_block(x, nb_filter[i])
    #--------------- center ------------
    i += 1


    #3
    x = up_conv(x, x3, nb_filter[i])
    # x = res_block(x, nb_filter[i])
    x = res_block(x, nb_filter[i])
    i += 1

    #2
    x = up_conv(x, x2, nb_filter[i])
    # x = res_block(x, nb_filter[i])
    x = res_block(x, nb_filter[i])
    i += 1

    #1
    x = up_conv(x, x1, nb_filter[i])
    # x = res_block(x, nb_filter[i])
    x = res_block(x, nb_filter[i])
    i += 1

    #0
    x = up_conv(x, x0, nb_filter[i])
    # x = res_block(x, nb_filter[i])
    x = res_block(x, nb_filter[i])
    x = Activation('relu')(x)

    classify = Conv2D(1, (1, 1), activation='sigmoid')(x)
    model = Model(inputs=inputs, outputs=classify)


    model.compile(optimizer = Adam(lr = 1e-4),
            loss = binary_crossentropy,
            metrics = [dice_coef])

    # model.compile(
    #         optimizer=RMSprop(lr=lr),
    #         loss=bce_dice_loss,
    #         metrics=[dice_loss])

    return model

def getSEUnet():
    model = create_model((256,256,1))
    #print(model.summary())
    return model
