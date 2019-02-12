#import Mask RCNN
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
MODEL_DIR = os.path.join('../logs/', "mrcnn_logs")
model = modellib.MaskRCNN(mode='training', config=config,
        model_dir=MODEL_DIR)
