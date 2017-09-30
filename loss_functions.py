from keras.preprocessing.image import ImageDataGenerator
import json
import os
from keras import backend as K

with open(os.getcwd() + '/train_config.json') as data_file:
    parameters = json.load(data_file)

    def soft_sorensen_dice(y_true, y_pred, axis=None, smooth=1):
        intersection = K.sum(y_true * y_pred, axis=axis)
        area_true = K.sum(y_true, axis=axis)
        area_pred = K.sum(y_pred, axis=axis)
        return (2 * intersection + smooth) / (area_true + area_pred + smooth)

    def hard_sorensen_dice(y_true, y_pred, axis=None, smooth=1):
        y_true_int = K.round(y_true)
        y_pred_int = K.round(y_pred)
        return soft_sorensen_dice(y_true_int, y_pred_int, axis, smooth)

    def dice(y_true, y_pred):
        batch_dice_coefs = hard_sorensen_dice(y_true, y_pred, axis=[1, 2])
        dice_coefs = K.mean(batch_dice_coefs, axis=0)
        return dice_coefs[1]
