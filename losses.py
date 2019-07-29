import keras.backend as K
from keras.losses import binary_crossentropy

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


