from keras.models import Model
from keras.layers import Input, MaxPooling2D, Dropout, Conv2D, Conv2DTranspose, TimeDistributed, Bidirectional, ConvLSTM2D
from keras import backend as K
import tensorflow as tf
from keras.optimizers import RMSprop, Adam, SGD
from keras.losses import binary_crossentropy
from losses import *
import math

from datahandler import DataHandler
from models.unet_se import *
from models.unet import *
from models.resnet_fcn import *
from models.resnet_se_fcn import *
from models.resnet_fcn import *
from models.vgg19_fcn import *
from models.vgg19_se_fcn import *
from models.unet_resnet import *

from generator import *
from params import *
from callbacks import getCallbacks
from data_loader import *

from tqdm import tqdm
import os
import skimage.io as io

from keras.models import *
from keras import backend as K

import argparse
import sys
import random


def getModel(name):
    if model_type == 'unet':
        print('using unet as first model'
        model = getUnet()
    elif model_type == 'resnetFCN':
        print('using resnetFCN as first model'
        model = getResnet50FCN()
    elif model_type == 'resnetSEFCN':
        print('using resnetSEFCN as first model'
        model = getResnetSE50FCN()
    elif model_type == 'vgg19FCN':
        print('using vgg19FCN as first model'
        model = getVGG19FCN()
    elif model_type == 'vgg19SEFCN':
        print('using vgg19SEFCN as first model'
        model = getVGG19SEFCN()
    elif model_type == 'UnetResNet18':
        print('using UnetResNet18 as first model'
        model = getUnetResnet18()
    elif model_type == 'UnetResNet18SE':
        print('using UnetResNet18SE as first model'
        model = getUnetResnet18(se_version = True)
    else:
        print('using UnetResNet18SE as first model'
        model = getSEUnet()

    return model


def lstmGenerator(images, masks, batch_size, pre_model, pre_graph):
    i=0
    while True:
        with pre_graph.as_default():
            batch_features = []
            batch_labels = []
            c = batch_size * i

            if c >= len(images) - 1:
                i = 0
                c = 0
            for j in range(c, c + batch_size):
                if j == 0:
                    res1 =  np.expand_dims(np.zeros(images[j].shape), axis=0)
                else:
                    img1 = np.expand_dims(images[j-1], axis=0)
                    res1 = pre_model.predict(img1)

                img2 = np.expand_dims(images[j], axis=0)
                res2 = pre_model.predict(img2)

                if j == images.shape[0]-1:
                    res3 = np.expand_dims(np.zeros(images[j].shape), axis=0)
                else:
                    img3 = np.expand_dims(images[j+1], axis=0)
                    res3 = pre_model.predict(img3)

                res = np.concatenate((res1,res2,res3), axis=0)
                res[res>=0.7] = 1
                res[res<0.7] = 0

                mask = masks[j]
                mask[mask == 255] = 1

                batch_features.append(res)
                batch_labels.append(mask)

            i += 1
            yield np.array(batch_features), np.array(batch_labels)

def lstmModel():

    with lstm_graph.as_default():

        inputs = Input((3, 256, 256, 1))

        bclstm = Bidirectional(ConvLSTM2D(32, 3, return_sequences = True, padding='same', activation = 'relu'))(inputs)
        bclstm = Bidirectional(ConvLSTM2D(32, 3, return_sequences = True, padding='same', activation = 'relu'))(bclstm)

        pool = TimeDistributed(MaxPooling2D(pool_size=2))(bclstm)

        bclstm = Bidirectional(ConvLSTM2D(64, 3, return_sequences = True, padding='same', activation = 'relu'))(pool)
        bclstm = Bidirectional(ConvLSTM2D(64, 3, return_sequences = True, padding='same', activation = 'relu'))(bclstm)
        bclstm = Bidirectional(ConvLSTM2D(64, 3, padding='same', activation = 'relu'))(bclstm)

        up = Conv2DTranspose(64,3, strides=2, padding='same', activation = 'relu')(bclstm)
        conv = Conv2D(64, 3, activation = 'relu', padding='same')(up)

        outputs = Conv2D(1, (1,1), activation = 'sigmoid')(conv)

        model = Model(input = inputs, output = outputs)

        model.compile(optimizer = Adam(lr = 1e-4),
                loss = binary_crossentropy, metrics = [dice_coef])

        return model

model_names = ['unet'] #['vgg19SEFCN', 'resnetFCN', 'resnetSEFCN', 'unetResnet18', 'unetResnet18SE']


for model_type in model_names:
    K.clear_session()
    print('Working with %s'%model_type)
    image_files, mask_files = load_data_files('data/kfold_data/')

    skf = getKFolds(image_files, mask_files, n=10)

    kfold_indices = []
    for train_index, val_index in skf.split(image_files, mask_files):
        kfold_indices.append({'train': train_index, 'val': val_index})

    #Get data and generators
    dh = DataHandler()

    pre_graph = tf.get_default_graph()

    for i in range(len(kfold_indices)):
        with pre_graph.as_default():
            pre_model = getModel(model_type)
            pre_model.load_weights('logs/unet/kfold_unet/kfold_unet_dice_DA_K%d/kfold_unet_dice_DA_K%d_weights.h5'%(i,i))

        exp_name = 'kfold_%s_dice_LSTM_K%d'%(model_type, i)
        #get parameters
        params = getParams(exp_name, model_type, is_lstm=True)

        #set common variables
        epochs = 10
        batch_size = 2
        verbose = 1

        tr_images, tr_masks, te_images, te_masks = dh.getKFoldData(image_files,
                mask_files, kfold_indices[i])

        train_generator = lstmGenerator(tr_images, tr_masks, batch_size, pre_model, pre_graph)
        val_generator = lstmGenerator(te_images, te_masks, batch_size, pre_model, pre_graph)

        #Get model and add weights

        lstm_graph = tf.get_default_graph()
        with lstm_graph.as_default():
            model = lstmModel()

        model_json = model.to_json()
        with open(params['model_name'], "w") as json_file:
             json_file.write(model_json)

        Checkpoint, EarlyStop, ReduceLR, Logger, TenBoard = getCallbacks(params)

        #Train the model
        with lstm_graph.as_default():
            history = model.fit_generator(train_generator,
                epochs=epochs,
                steps_per_epoch = len(tr_images) / batch_size,
                validation_data = val_generator,
                validation_steps = len(te_images) / batch_size,
                verbose = verbose,
                max_queue_size = 1,
                callbacks = [Checkpoint, TenBoard])
