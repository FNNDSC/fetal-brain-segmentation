from keras import layers

from keras.applications.vgg19 import VGG19
from keras.models import Model

def getVGG19FCN():
	# load ResNet
	n_classes = 1
	stride = 32

	input_tensor = layers.Input(shape=(256, 256, 1))
	base_model = VGG19(weights=None, include_top=False, input_tensor=input_tensor)

	# add classifier
	x = base_model.get_layer('block5_pool').output
	x = layers.Dropout(0.5)(x)

	x = layers.Conv2D(n_classes,1,name = 'pred_32',init='zero',padding = 'valid')(x)

	## add 32s upsampler

	x = layers.UpSampling2D(size=(stride), interpolation='bilinear')(x)
	x = layers.Activation('sigmoid')(x)
	pred_32s = x

	# 16s
	x = base_model.get_layer('block5_conv4').output
	x = layers.Dropout(0.5)(x)
	x = layers.Conv2D(n_classes,1,name = 'pred_16',init='zero',padding = 'valid')(x)
	x = layers.UpSampling2D(name='upsampling_16',size=(stride//2), interpolation='bilinear')(x)
	x = layers.Conv2D(n_classes,5,name = 'pred_up_16',init='zero',padding = 'same')(x)

	# merge classifiers
	x = layers.add([x, pred_32s])
	x = layers.Activation('sigmoid')(x)
	pred_16s = x

	x = base_model.get_layer('block4_conv4').output
	x = layers.Dropout(0.5)(x)
	x = layers.Conv2D(n_classes,1,name = 'pred_8',init='zero',padding = 'valid')(x)
	x = layers.UpSampling2D(name='upsampling_8',size=(stride//4), interpolation='bilinear')(x)
	x = layers.Conv2D(n_classes,5,name = 'pred_up_8',init='zero',padding = 'same')(x)

	# merge classifiers
	x = layers.add([x, pred_16s])
	x = layers.Activation('sigmoid')(x)

	model = Model(input=base_model.input,output=x)
	print(model.summary())

	return model
