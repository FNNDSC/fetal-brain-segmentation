import keras.backend as K
from keras.losses import binary_crossentropy
import tensorflow as tf

### DICE
def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

### BORDER
def get_border(y_true):
    mask = (25,25)
    pos = y_true
    neg = 1-y_true

    pos = K.pool2d(pos, pool_size=mask, padding='same')
    neg = K.pool2d(neg, pool_size=mask, padding='same')

    border = pos * neg
    return border

def border_dice_coef(y_true, y_pred):
    border = get_border(y_true)
    flat_border = K.flatten(border)
    flat_y_true = K.flatten(y_true)
    flat_y_pred = K.flatten(y_pred)

    border_y_true = K.tf.gather(flat_y_true, K.tf.where(flat_border > 0.5))
    border_y_pred = K.tf.gather(flat_y_pred, K.tf.where(flat_border > 0.5))

    return dice_coef(border_y_true, border_y_pred)

def border_dice_loss(y_true, y_pred):
    return 1 - border_dice_coef(y_true, y_pred)

### DICE + BORDER LOSS
def dice_border_loss(y_true, y_pred):
    return (border_dice_loss(y_true, y_pred) * 0.05 +
            (dice_loss(y_true, y_pred) * 0.95))

### BCE
def bce_dice_loss(y_true, y_pred):
    return 0.5 * binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)

### FOCAL
def binary_focal_loss(gamma=2., alpha=.25):
    """
    Binary form of focal loss.
      FL(p_t) = -alpha * (1 - p_t)**gamma * log(p_t)
      where p = sigmoid(x), p_t = p or 1 - p depending on if the label is 1 or 0, respectively.
    References:
        https://arxiv.org/pdf/1708.02002.pdf
    Usage:
     model.compile(loss=[binary_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = K.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
        pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)

        return -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))


    return binary_focal_loss_fixed

