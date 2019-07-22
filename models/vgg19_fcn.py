from keras import layers
from keras.applications.vgg19 import VGG19
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K
from keras.losses import binary_crossentropy
import tensorflow as tf

from losses import *

def getVGG19FCN():
	n_classes = 1
	stride = 32

	input_tensor = layers.Input(shape=(256, 256, 1))
	base_model = VGG19(weights=None, include_top=False, input_tensor=input_tensor)

	#32
	pool_5 = base_model.get_layer('block5_pool').output
	up_32 = layers.Conv2DTranspose(n_classes, 3, name='up_32', strides=(stride), activation='relu', kernel_initializer = 'he_normal')(pool_5)
	pred_32 = layers.Conv2D(n_classes, 3, name='pred_32', padding = 'same', activation='sigmoid', kernel_initializer = 'he_normal')(up_32)

	#16
	pool_4 = base_model.get_layer('block4_pool').output
	up_16 = layers.Conv2DTranspose(n_classes, 3, name='up_16', strides=(stride//2), activation='relu', kernel_initializer = 'he_normal')(pool_4)
	addition_1 = layers.concatenate([up_16, pred_32])
	pred_16 = layers.Conv2D(n_classes, 3, name='pred_16', padding = 'same', activation='sigmoid', kernel_initializer = 'he_normal')(addition_1)

	#8
	pool_3 = base_model.get_layer('block3_pool').output
	up_8 = layers.Conv2DTranspose(n_classes, 3, name='up_8', strides=(stride//4), activation='relu', kernel_initializer = 'he_normal')(pool_3)
	addition_2 = layers.concatenate([up_8, pred_16])
	pred_8 = layers.Conv2D(n_classes, 3, name='pred_8', padding = 'same', activation='sigmoid', kernel_initializer = 'he_normal')(addition_2)

	x = pred_8

	model = Model(input=base_model.input,output=x)
	model.compile(optimizer = Adam(lr = 1e-4),
            loss = binary_crossentropy,
            metrics = [dice_coef])

	return model