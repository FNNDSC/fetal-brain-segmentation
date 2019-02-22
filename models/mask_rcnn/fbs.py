import os
import sys
import glob
import tqdm
import skimage.color

ROOT_DIR = os.path.abspath('../../')
sys.path.append(ROOT_DIR)

import numpy as np
import nibabel as nib
import skimage.io as io
from imgaug import augmenters as iaa

from callbacks import getCallbacks
from params import getParams

from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, 'logs')
DEFAULT_MODEL_DIR = os.path.join(DEFAULT_LOGS_DIR, 'mask_rcnn')

class FBSConfig(Config):
    NAME = 'FBS_RESNET50_DA_TF_WEIGHTEDLOSS_'

    IMAGES_PER_GPU = 4

    NUM_CLASSES = 1 + 1 # background + brain

    STEPS_PER_EPOCH = 404 * 4
    VALIDATION_STEPS = 116 * 4

    BACKBONE = 'resnet50'

    RPN_ANCHOR_SCALES = (16, 32, 64, 128)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Dont exclude based on confidence, since we have two classes
    # then 0.5 is the minimum anyway as it picks between brain and BG
    DETECTION_MIN_CONFIDENCE = 0

    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    MEAN_PIXEL = np.array([45.6, 45.6, 45.6])
    TRAIN_BN = None

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 100

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 1

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 1

    LOSS_WEIGHTS = {
        "rpn_class_loss": 0.,
        "rpn_bbox_loss": 0.2,
        "mrcnn_class_loss": 0.,
        "mrcnn_bbox_loss": 0.2,
        "mrcnn_mask_loss": 1.
    }

class FBSInferenceConfig(FBSConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7

#######################################################################
#   Data
#######################################################################

class FBSDataset(utils.Dataset):
    """
    Reads the dataset images and masks, and prepares them
    for the model
    """
    # values must be between 0 and 255
    def __normalize0_255(self, img_slice):
        rows, cols = img_slice.shape
        new_img = np.zeros((rows, cols))
        max_val = np.max(img_slice)

        for i in range(rows):
            for j in range(rows):
                new_img[i,j] = int((
                    float(img_slice[i,j])/float(max_val))
                    * 255)
        return new_img

    def load_data(self, dataset_dir, subset='train'):

        if subset is not 'train' and subset is not 'validate':
            print('Subset must be train or validate')
            return -1

        images_dir, masks_dir = None, None

        if subset is 'train':
            images_dir = os.path.join(dataset_dir, 'train/images/*')
            masks_dir = os.path.join(dataset_dir, 'train/masks/*')

        elif subset is 'validate':
            images_dir = os.path.join(dataset_dir, 'test/images/*')
            masks_dir = os.path.join(dataset_dir, 'test/masks/*')

        self.add_class('MRI', 1, 'brain')

        image_glob = sorted(glob.glob(images_dir))
        mask_glob = sorted(glob.glob(masks_dir))
        img_id = 0
        for i in tqdm.trange(len(image_glob), desc='loading data'):
            img_path = image_glob[i]
            mask_path = mask_glob[i]

            img_slices = nib.load(image_glob[i])
            mask_slices = nib.load(mask_glob[i])
            img_slices = img_slices.get_fdata()
            mask_slices = mask_slices.get_fdata()

            for j in range(img_slices.shape[-1]):
                img = np.array(img_slices[:,:,j])
                mask = np.array(mask_slices[:,:,j])

                # skip images that are not 256x256
                if img.shape[0] != 256 or img.shape[1] != 256:
                    break

                # Normalize image so its between 0-255
                new_img = self.__normalize0_255(img)
                new_img = skimage.color.gray2rgb(new_img)

                # img = new_img[..., np.newaxis]
                mask = mask[..., np.newaxis]

                mask = np.array(mask, dtype=np.uint16) * 255
                new_img = np.array(new_img, dtype=np.uint16)

                self.add_image('MRI',
                        image_id=img_id,
                        path=img_path,
                        mask_path=mask_path,
                        image=new_img,
                        mask=mask,
                        width=256,
                        height=256)

                img_id += 1

    def load_image(self, image_id):
        return self.image_info[image_id]['image']

    def load_mask(self, image_id):
        mask = self.image_info[image_id]['mask']
        return mask, np.ones([1]).astype(np.int32)

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        return self.image_info[image_id]

def train(dataset_dir, augment = False,
        pretrained_coco = False):
    """Train the model"""

    config = FBSConfig()

    dataset_train = FBSDataset()
    dataset_train.load_data(dataset_dir, subset='train')
    dataset_train.prepare()

    dataset_val = FBSDataset()
    dataset_val.load_data(dataset_dir, subset='validate')
    dataset_val.prepare()

    augmentation = None
    if augment:
        augmentation = iaa.SomeOf((0, 6), [
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.OneOf([iaa.Affine(rotate=90),
                iaa.Affine(rotate=180),
                iaa.Affine(rotate=270)]),
            iaa.Multiply((0.8, 1.5)),
            iaa.GaussianBlur(sigma=(0.0, 2.0))])

    model = modellib.MaskRCNN(mode='training', config=config,
            model_dir=DEFAULT_MODEL_DIR)

    params = getParams('Mask_RCNN')
    epochs = params['epochs']

    #TODO remove useless callbacks
    _, EarlyStop, _, _, _ = getCallbacks(params)

    if pretrained_coco:
        COCO_MODEL_PATH = 'mask_rcnn_coco.h5'
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                    "mrcnn_bbox", "mrcnn_mask"])

    model.train(dataset_train, dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs = epochs,
        augmentation = augmentation,
        custom_callbacks = [EarlyStop],
        layers = 'all')


