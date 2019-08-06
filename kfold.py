from datahandler import DataHandler

from models.unet_se import *
from models.unet import *
from models.resnet_se_fcn import *
from models.unet_resnet_se import *
from models.resnet_fcn import *
from models.vgg19_fcn import *
from models.vgg19_se_fcn import *
from models.unet_upconv import *
from models.unet_upconv_bn import *
from models.unet_upconv_se import *
from models.unet_resnet_upconv_se import *

from models.unet_attention import *
from models.vgg19_attention import *
from models.vgg19_fcn_upconv import *

from models.unet_f_attention import *
from models.unet_f_g_attention import *

from models.unet_bn import *

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
import tensorflow as tf

def getModel(name):
    print('Working with %s'%name)

    if name == 'unet_upconv':
        model = getUnetUpconv()

    elif name == 'unet_upconv_bn':
        model = getUnetUpconvBN()

    elif name == 'unet_upconv_se':
        model = getSEUnetUpconv()

    elif name == 'resnetFCN':
        model = getResnet50FCN()
    elif name == 'resnetSEFCN':
        model = getResnetSE50FCN()

    elif name == 'vgg19FCN':
        model = getVGG19FCN()
    elif name == 'vgg19SEFCN':
        model = getVGG19SEFCN()

    elif name == 'unet_resnet_upconv':
        model = getUnetResUpconv()
    elif name == 'unet_resnet_upconv_se':
        model = getUnetResUpconv(se_version = True)

    elif name == 'unet_attention':
        model = getAttentionUnet()

    elif name == 'vgg19FCN_attention':
        model = getVGG19Attention()

    elif name == 'vgg19_fcn_upconv':
        model = getVGG19FCN_upconv()

    elif name == 'unet_filter_attention':
        model = getUnetFilterAttention()

    elif name == 'unet_filter_grid_attention':
        model = getUnetFilterGridAttention()

    elif name == 'unet_bn':
        model = getUnetBN('BCE')

    elif name == 'unet_bn_dice_loss':
        model = getUnetBN('dice')

    elif name == 'unet_bn_focal_loss':
        model = getUnetBN('focal')

    elif name == 'unet_bn_bce_dice_loss':
        model = getUnetBN('BCE_DICE')

    # elif name == 'unet_resnet_upconv':
    #     model = getUnetResUpconv()
    # elif name == 'unet_resnet_upconv_se':
    #     model = getUnetResUpconv(se_version = True)
    # elif name == 'unetResnet18':
    #     model = getUnetResnet18()
    # elif name == 'unetResnet18SE':
    #     model = getUnetResnet18(se_version = True)
    else:
        print('error')
        return -1

    return model

model_names = ['unet_filter_attention', 'unet_filter_grid_attention', 'unet_bn',
                'unet_bn_dice_loss', 'unet_bn_focal_loss', 'unet_bn_bce_dice_loss']

# model_names = ['unet_upconv', 'unet_upconv_se',
        # 'unet_resnet_upconv', 'unet_resnet_upconv_se']

for model_type in model_names:
    image_files, mask_files = load_data_files('data/kfold_data/')

    skf = getKFolds(image_files, mask_files, n=10)

    kfold_indices = []
    for train_index, val_index in skf.split(image_files, mask_files):
        kfold_indices.append({'train': train_index, 'val': val_index})

    #Get data and generators
    dh = DataHandler()

    start = 0
    end = len(kfold_indices)

    if model_type == 'unet_filter_attention':
        start = 7

    for i in range(start, end):

        exp_name = 'kfold_%s_dice_DA_K%d'%(model_type, i)

        #get parameters
        params = getParams(exp_name, model_type)

        #set common variables
        epochs = params['epochs']
        batch_size = params['batch_size']
        verbose = params['verbose']

        #Get model and add weights
        model = getModel(model_type)


        tr_images, tr_masks, te_images, te_masks = dh.getKFoldData(image_files,
                mask_files, kfold_indices[i])

        train_generator = getGenerator(tr_images, tr_masks,
                augmentation = False, batch_size=batch_size)
        val_generator = getGenerator(te_images, te_masks,
                augmentation = False, batch_size=batch_size)

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

