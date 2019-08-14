from keras.models import Model
from keras.layers import Input, MaxPooling2D, Dropout, Conv2D, Conv2DTranspose, TimeDistributed, Lambda, Bidirectional, ConvLSTM2D, add
from keras import backend as K
import tensorflow as tf
from keras.optimizers import RMSprop, Adam, SGD
from keras.losses import binary_crossentropy
from losses import *
import math

from datahandler import DataHandler

from model_provider import getModel

from generator import *
from params import *
from callbacks import getCallbacks
from kfold_data_loader import *

from tqdm import tqdm
import os
import skimage.io as io

from keras.models import *
from keras import backend as K

import argparse
import sys
import random

def lstmGenerator(images, masks, batch_size, pre_model, pre_graph):
    reset = False

    while True:
        with pre_graph.as_default():
            batch_features = []
            batch_labels = []

            for i in range(batch_size):

                j = np.random.choice(len(images),1)[0]

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
                res[res>=0.5] = 1
                res[res<0.5] = 0

                mask = masks[j]
                mask[mask == 255] = 1

                batch_features.append(res)
                batch_labels.append(mask)


            yield np.array(batch_features), np.array(batch_labels)

def lstmModel():

    with lstm_graph.as_default():

        inputs = Input((3, 256, 256, 1))

        original = Lambda(lambda x : x[:,1,:,:,:] * 0.5)(inputs)

        pool = TimeDistributed(MaxPooling2D(pool_size=2))(inputs)
        bclstm = Bidirectional(ConvLSTM2D(64, 3, return_sequences = True,
                                          padding='same', activation = 'relu'))(pool)
        bclstm = Bidirectional(ConvLSTM2D(64, 3, padding='same', activation = 'relu'))(bclstm)

        up = Conv2DTranspose(64,3, strides=2, padding='same', activation = 'relu')(bclstm)
        drop = Dropout(0.5)(up)
        outputs = Conv2D(1, (1,1), activation = 'sigmoid')(drop)

        outputs = Lambda(lambda x : x * 0.5)(outputs)

        outputs = add([outputs, original])

        model = Model(inputs = inputs, outputs = outputs)

        model.compile(optimizer = Adam(lr = 1e-4),
                loss = binary_crossentropy, metrics = [dice_coef])

        return model

model_type = 'unet'

K.clear_session()

image_files, mask_files = load_data_files('data/kfold_data/')

skf = getKFolds(image_files, mask_files, n=10)

kfold_indices = []
for train_index, val_index in skf.split(image_files, mask_files):
    kfold_indices.append({'train': train_index, 'val': val_index})

#Get data and generators
dh = DataHandler()

pre_graph = tf.get_default_graph()


for i in range(9,10):
    with pre_graph.as_default():
        pre_model = getModel(model_type)
        pre_model.load_weights('logs/%s/kfold_%s/kfold_%s_dice_DA_K%d/kfold_%s_dice_DA_K%d_weights.h5'%(
            model_type,model_type,model_type,i,model_type,i))

    exp_name = 'kfold_%s_BiCLSTM_K%d'%(model_type, i)
    #get parameters
    params = getParams(exp_name, model_type, is_lstm=True)

    #set common variables
    epochs = 10
    batch_size = 10
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
            steps_per_epoch = 200,
            validation_data = val_generator,
            validation_steps = 20,
            verbose = verbose,
            max_queue_size = 1,
            callbacks = [Checkpoint, TenBoard])
