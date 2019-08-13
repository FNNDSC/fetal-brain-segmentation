from mrcnn.config import Config
import numpy as np
import math

class FBSConfig(Config):
    NAME = 'FBM' #Overwrite

    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 1 # background + brain

    BACKBONE = 'resnet50'

    IMAGE_CHANNEL_COUNT = 1

    RPN_ANCHOR_SCALES = (16, 32, 64, 128)

    RPN_ANCHOR_RATIOS = [0.75, 1, 1.5]

    # Dont exclude based on confidence, since we have two classes
    # then 0.5 is the minimum anyway as it picks between brain and BG
    DETECTION_MIN_CONFIDENCE = 0

    USE_MINI_MASK = True

    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 256
    IMAGE_MAX_DIM = 256

    IMAGE_SHAPE = [256,256,1]

    # one channel
    MEAN_PIXEL = np.array([73.99])

    TRAIN_BN = False

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 1

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 1

    def __init__(self):
        super(Config, self).__init__()

class TrainFBSConfig(FBSConfig):
    def __init__(self,
            base_name = 'FBM_resnet50',
            da = False,
            tl = False,
            wl = False,
            mask_dim = 28,
            kfold_i = 0,
            img_per_gpu=2,
            train_steps=1000,
            val_steps=50,
            epochs=25,
            n_folds=10):


        if da:
            base_name += '_da'
        if tl:
            base_name += '_tl'
        if wl:
            self.LOSS_WEIGHTS={
                    'rpn_class_loss': 0.5,
                    'rpn_bbox_loss': 0.5,
                    'mrcnn_class_loss': 0.5,
                    'mrcnn_bbox_loss': 0.5,
                    'mrcnn_mask_loss': 1.
                    }
            base_name += '_wl'
        base_name += '_%d'%epochs
        base_name += '_%d'%mask_dim
        base_name += '_%dK_%d_'%(n_folds, kfold_i)

        self.TRAIN_ROIS_PER_IMAGE = 100
        self.NAME = base_name
        self.IMAGES_PER_GPU = img_per_gpu

        self.STEPS_PER_EPOCH = math.ceil(train_steps / img_per_gpu)
        self.VALIDATION_STEPS = math.ceil(val_steps / img_per_gpu)


        super(FBSConfig, self).__init__()

class InferenceFBSConfig(TrainFBSConfig):
    def __init__(self, **args):
        super(InferenceFBSConfig, self).__init__(**args)
        self.GPU_COUNT = 1
        self.IMAGES_PER_GPU = 1
        super(FBSConfig, self).__init__()


