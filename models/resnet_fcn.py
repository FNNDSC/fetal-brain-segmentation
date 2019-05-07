from keras.layers import Input, Dropout, add
from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers.core import Activation

from keras.applications.resnet50 import ResNet50
from keras.models import Model


# load ResNet
n_classes = 1
stride = 32

input_tensor = Input(shape=(256, 256, 1))
base_model = ResNet50(weights=None, include_top=False, input_tensor=input_tensor)

# add classifier
x = base_model.get_layer('activation_49').output
x = Dropout(0.5)(x)

x = Convolution2D(n_classes,1,1,name = 'pred_32',init='zero',border_mode = 'valid')(x)

# add 32s upsampler

x = UpSampling2D(size=(stride), interpolation='bilinear')(x)
x = Activation('sigmoid')(x)
pred_32s = x

# 16s
x = base_model.get_layer('activation_40').output
x = Dropout(0.5)(x)
x = Convolution2D(n_classes,1,1,name = 'pred_16',init='zero',border_mode = 'valid')(x)
x = UpSampling2D(name='upsampling_16',size=(stride//2), interpolation='bilinear')(x)
x = Convolution2D(n_classes,5,5,name = 'pred_up_16',init='zero',border_mode = 'same')(x)

# merge classifiers
x = add([x, pred_32s])
x = Activation('sigmoid')(x)
pred_16s = x

x = base_model.get_layer('activation_22').output
x = Dropout(0.5)(x)
x = Convolution2D(n_classes,1,1,name = 'pred_8',init='zero',border_mode = 'valid')(x)
x = UpSampling2D(name='upsampling_8',size=(stride//4), interpolation='bilinear')(x)
x = Convolution2D(n_classes,5,5,name = 'pred_up_8',init='zero',border_mode = 'same')(x)

# merge classifiers
x = add([x, pred_16s])
x = Activation('sigmoid')(x)

model = Model(input=base_model.input,output=x)
print(model.summary())