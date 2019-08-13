import os
import sys
import numpy as np

from data_loader import *
from fbs_config import TrainFBSConfig, InferenceFBSConfig
from fbs_dataset import FBSDataset

from imgaug import augmenters as iaa
from mrcnn import model as modellib

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

ROOT_DIR = os.path.abspath('../../../')
sys.path.append(ROOT_DIR)

DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
DEFAULT_MODEL_DIR = os.path.join(DEFAULT_LOGS_DIR, 'mask_rcnn/kfold')
epochs = 25
n_folds = 10

def getDatasets(train_indices, val_indices):
    image_train_files = np.take(image_files, train_indices)
    mask_train_files = np.take(mask_files, train_indices)

    image_val_files = np.take(image_files, val_indices)
    mask_val_files = np.take(mask_files, val_indices)

    train_files = (image_train_files, mask_train_files)
    val_files = (image_val_files, mask_val_files)

    dataset_train = FBSDataset()
    len_dataset_train = dataset_train.load_data(train_files)
    dataset_train.prepare()

    dataset_val = FBSDataset()
    len_dataset_val = dataset_val.load_data(val_files)
    dataset_val.prepare()

    return dataset_train, len_dataset_train, dataset_val, len_dataset_val


def train(trainConfig, train_dataset, val_dataset, augment = True, pretrained_coco = True):
    augmentation = None

    if augment:
        augmentation = iaa.SomeOf((0, 6), [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Multiply((0.5, 1.5), per_channel=0.5),
            iaa.OneOf([iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270)])])

    model = modellib.MaskRCNN(mode='training', config=trainConfig,
            model_dir=DEFAULT_MODEL_DIR)


    if pretrained_coco:
        COCO_MODEL_PATH = '../mask_rcnn_coco.h5'
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                exclude=["conv1", "mrcnn_class_logits", "mrcnn_bbox_fc",
                    "mrcnn_bbox", "mrcnn_mask"])


    model.train(dataset_train, dataset_val,
            learning_rate=trainConfig.LEARNING_RATE,
            epochs = epochs,
            augmentation = augmentation,
            save_best_only = True,
            monitored_quantity = 'val_mrcnn_mask_loss',
            layers = 'all')


image_files, mask_files = load_data_files('data/kfold_data/')

skf = getKFolds(image_files, mask_files, n=n_folds)

kfold_indices = []
for train_index, test_index in skf.split(image_files, mask_files):
    kfold_indices.append({'train': train_index, 'val': test_index})

for i in range(5, 10):
    dataset_train, len_dataset_train, dataset_val, len_dataset_val = getDatasets(
            kfold_indices[i]['train'], kfold_indices[i]['val'])

    configParams = {'da': True, 'mask_dim': 28, 'wl': True, 'tl': True, 'kfold_i': i,
            'img_per_gpu': 2, 'train_steps': len_dataset_train,
            'val_steps': len_dataset_val, 'epochs': epochs, 'n_folds': n_folds}

    trainFBSConfig = TrainFBSConfig(**configParams)
    trainFBSConfig.display()

    train(trainFBSConfig, dataset_train, dataset_val,
            augment=configParams['da'], pretrained_coco=configParams['tl'])



