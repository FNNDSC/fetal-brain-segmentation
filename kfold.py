from datahandler import DataHandler
from models.unet_se import *
from models.unet import *
from models.resnet_fcn import *
from models.resnet_se_fcn import *
from models.resnet_fcn import *
from models.vgg19_fcn import *
from models.vgg19_se_fcn import *

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

def getModel(name):
    if model_type == 'unet':
        model = getUnet()
    elif model_type == 'resnetFCN':
        model = getResnet50FCN()
    elif model_type == 'resnetSEFCN':
        model = getResnetSE50FCN()
    elif model_type == 'vgg19FCN':
        model = getVGG19FCN()
    elif model_type == 'vgg19SEFCN':
        model = getVGG19SEFCN()
    else:
        model = getSEUnet()

    return model

model_names = ['vgg19FCN', 'vgg19SEFCN', 'resnetFCN', 'resnetSEFCN']

for model_type in model_names:
    print('Working with %s'%model_type)
    image_files, mask_files = load_data_files('data/kfold_data/')

    skf = getKFolds(image_files, mask_files, n=10)

    kfold_indices = []
    for train_index, val_index in skf.split(image_files, mask_files):
        kfold_indices.append({'train': train_index, 'val': val_index})

    #Get data and generators
    dh = DataHandler()
    for i in range(len(kfold_indices)):

        exp_name = 'kfold_%s_dice_DA_K%d'%(model_type, i)

        #get parameters
        params = getParams(exp_name, model_type)

        #set common variables
        epochs = params['epochs']
        batch_size = params['batch_size']
        verbose = params['verbose']

        tr_images, tr_masks, te_images, te_masks = dh.getKFoldData(image_files,
                mask_files, kfold_indices[i])

        train_generator = getGenerator(tr_images, tr_masks,
                augmentation = False, batch_size=batch_size)
        val_generator = getGenerator(te_images, te_masks,
                augmentation = False, batch_size=batch_size)

        #Get model and add weights
        model = getModel(model_type)

        model_json = model.to_json()
        with open(params['model_name'], "w") as json_file:
             json_file.write(model_json)

        Checkpoint, EarlyStop, ReduceLR, Logger, TenBoard = getCallbacks(params)

        #Train the model
        history = model.fit_generator(train_generator,
                epochs=epochs,
                steps_per_epoch = len(tr_images) / batch_size,
                validation_data = val_generator,
                validation_steps = len(te_images) / batch_size,
                verbose = verbose,
                callbacks = [Checkpoint, Logger, TenBoard])
