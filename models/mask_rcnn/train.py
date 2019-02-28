import os
import sys

from fbs import train

ROOT_DIR = os.path.abspath('../../')
sys.path.append(ROOT_DIR)

DATASET_DIR = os.path.join(ROOT_DIR, 'data/')

train(DATASET_DIR,
        augment=False,
        pretrained_coco=True)
