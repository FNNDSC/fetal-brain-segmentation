import os
import sys
ROOT_DIR = os.path.abspath('.')
sys.path.append(ROOT_DIR)

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
import tensorflow as tf

#list of model names you want to train,
#the logs will be saved with these names
model_names = ['unet']

for model_type in model_names:
    #load data files and split into 10-fold        
    image_files, mask_files = load_data_files('data/kfold_data/')
    skf = getKFolds(image_files, mask_files, n=10)

    kfold_indices = []
    for train_index, val_index in skf.split(image_files, mask_files):
        kfold_indices.append({'train': train_index, 'val': val_index})

    #Get data and generators
    dh = DataHandler()

    start = 0
    end = len(kfold_indices)

    for i in range(start, end):

        #create the experiment name for each subfolder
        exp_name = 'kfold_%s_dice_DA_K%d'%(model_type, i)

        #get parameters
        params = getParams(exp_name, model_type)

        #set common variables
        epochs = params['epochs']
        batch_size = params['batch_size']
        verbose = params['verbose']
        augmentation = False

        steps_per_epoch = len(tr_images) / batch_size

        #This I used because using augmentation I double the number of trainig data,
        #but I also increase de batch size
        if 'unet_bn' in model_type or 'unet_attention_bn' in model_type:
            batch_size *= 2
            augmentation = True
            steps_per_epoch = 2 * len(tr_images) / batch_size

        #Get model and add weights
        model = getModel(model_type)

        #get the data for each fold
        tr_images, tr_masks, te_images, te_masks = dh.getKFoldData(image_files,
                mask_files, kfold_indices[i])

        #get generators
        train_generator = getGenerator(tr_images, tr_masks,
                augmentation = augmentation, batch_size=batch_size)
        val_generator = getGenerator(te_images, te_masks,
                augmentation = False, batch_size=batch_size)

        #save model file        
        model_json = model.to_json()
        with open(params['model_name'], "w") as json_file:
             json_file.write(model_json)

        #get callbacks
        Checkpoint, EarlyStop, ReduceLR, Logger, TenBoard = getCallbacks(params)

        #Train the model
        history = model.fit_generator(train_generator,
                epochs=epochs,
                steps_per_epoch = steps_per_epoch,
                validation_data = val_generator,
                validation_steps = len(te_images) / batch_size,
                verbose = verbose,
                callbacks = [Checkpoint, Logger, TenBoard])