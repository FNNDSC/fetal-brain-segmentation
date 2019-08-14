from datahandler import DataHandler
from models.unet.unet import *
from generator import *
from params import *
from callbacks import getCallbacks

from tqdm import tqdm
import os
import skimage.io as io

from keras.models import *
from keras import backend as K

import argparse
import sys

#Get argument from command line
parser = argparse.ArgumentParser()
parser.add_argument('--exp', required=True,
    help='Experiments name, to save weights and logs')

args = parser.parse_args()
exp_name = args.exp

#get parameters
params = getParams(exp_name)

#set common variables
epochs = params['epochs']
batch_size = params['batch_size']
verbose = params['verbose']
val_to_monitor = params['val_to_monitor']

resetSeed()

#Get data and generators
dh = DataHandler()
tr_images, tr_masks, te_images, te_masks = dh.getData()

train_generator = getGenerator(tr_images, tr_masks,
        augmentation = False, batch_size=batch_size)
        #params['train_augmantation'], batch_size=batch_size)
val_generator = getGenerator(te_images, te_masks,
        augmentation = False, batch_size=batch_size)

#Get model and add weights
model = getUnet()

#load weights from other problem transfer learning
#model.load_weights('./weights/unet_transfer.h5')

# print(model.summary())

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
        callbacks = [Checkpoint, EarlyStop, ReduceLR, Logger, TenBoard])

