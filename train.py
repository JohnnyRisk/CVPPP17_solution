#!/usr/bin/env python

from __future__ import division, print_function

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
import json
import tensorflow as tf
from keras import metrics

import models, dataset

sub_dir = '/train'
l_masks = sub_dir + '/labeled_masks'
masks = sub_dir + '/masks'
centers = sub_dir + '/centers'
images = sub_dir + '/images'
start_dir = '/2017_LSC/training/A4'

train_img_dir = os.getcwd() + '/train' + '/images'
train_mask_dir = os.getcwd() + '/train' + '/masks'
val_img_dir = os.getcwd() + '/validation' + '/images'
val_mask_dir = os.getcwd() + '/validation' + '/masks'
test_img_dir = os.getcwd() + '/test' + '/images'
test_mask_dir = os.getcwd() + '/test' + '/masks'

from keras import losses, optimizers, utils
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.callbacks import ModelCheckpoint
from keras import backend as K


def select_optimizer(optimizer_name, optimizer_args):
    optimizers = {
        'sgd': SGD,
        'rmsprop': RMSprop,
        'adagrad': Adagrad,
        'adadelta': Adadelta,
        'adam': Adam,
        'adamax': Adamax,
        'nadam': Nadam,
    }
    if optimizer_name not in optimizers:
        raise Exception("Unknown optimizer ({}).".format(optimizer_name))

    return optimizers[optimizer_name](**optimizer_args)


def train():
    with open(os.getcwd() + '/train_config.json') as data_file:
        parameters = json.load(data_file)

    logging.basicConfig(level=logging.INFO)

    logging.info("Loading dataset...")
    augmentation_args = {
        'rotation_range': parameters["rotation_range"],
        'width_shift_range': parameters["width_shift_range"],
        'height_shift_range': parameters["height_shift_range"],
        'shear_range': parameters["shear_range"],
        'zoom_range': parameters["zoom_range"],
        'fill_mode': parameters["fill_mode"],
        'horizontal_flip': parameters["horizontal_flip"],
        'vertical_flip': parameters["vertical_flip"],
        'alpha': parameters["alpha"],
        'sigma': parameters["sigma"],
    }
    train_generator, train_steps_per_epoch, \
    val_generator, val_steps_per_epoch = dataset.create_generators(
        data_dir=os.getcwd(), batch_size=parameters["batch_size"],
        validation_split=parameters["validation_split"],
        shuffle_train_val=parameters["shuffle_train_val"],
        shuffle=parameters["shuffle"],
        seed=None,
        normalize_images=True,
        augment_training=parameters["augment_training"],
        augment_validation=parameters["augment_validation"],
        augmentation_args=augmentation_args)

    # get image dimensions from first batch
    images, masks = next(train_generator)
    _, height, width, channels = images.shape
    _, _, _, classes = masks.shape
    print("The mask shape is {}".format(masks.shape))
    print("The masks have the unique values of ...")
    print(np.unique(masks, return_counts=True))

    print("The images shape is {}".format(images.shape))
    print("The images unique vales of ...")
    print(np.unique(images[0, :, :, :], return_counts=True))

    logging.info("Building model...")
    """
    string_to_model = {
        "unet": models.unet,
        "dilated-unet": models.dilated_unet,
        "dilated-densenet": models.dilated_densenet,
        "dilated-densenet2": models.dilated_densenet2,
        "dilated-densenet3": models.dilated_densenet3,
    }
    """

    m = models.get_unet_8(height, width)

    m.summary()
    # instantiate optimizer, and only keep args that have been set
    # (not all optimizers have args like `momentum' or `decay')
    optimizer_args = {
        'lr': parameters['lr'],
        'momentum': None,
        'decay': None
    }
    for k in list(optimizer_args):
        if optimizer_args[k] is None:
            del optimizer_args[k]
    optimizer = select_optimizer('adam', optimizer_args)

    # select loss function: pixel-wise crossentropy, soft dice or soft
    # jaccard coefficient
    lossfunc = losses.binary_crossentropy

    def dice_coef(y_true, y_pred, axis=None, smooth=1):
        intersection = K.sum(y_true * y_pred, axis=axis)
        area_true = K.sum(y_true, axis=axis)
        area_pred = K.sum(y_pred, axis=axis)
        return (2 * intersection + smooth) / (area_true + area_pred + smooth)

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
        return dice_coefs

    metric = [metrics.binary_crossentropy, soft_sorensen_dice, hard_sorensen_dice]

    m.compile(optimizer=optimizer, loss=lossfunc, metrics=metric)

    # automatic saving of model during training
    filepath = os.path.join(
        os.getcwd(), 'checkpoints', "weights-{epoch:02d}-{val_soft_sorensen_dice:.4f}.hdf5")
    monitor = 'val_soft_sorensen_dice'
    mode = 'max'
    checkpoint = ModelCheckpoint(
        filepath, monitor=monitor, verbose=1,
        save_best_only=True, mode=mode)
    callbacks = [checkpoint]

    # train
    logging.info("Begin training.")
    m.fit_generator(train_generator,
                    epochs=parameters["epochs"],
                    steps_per_epoch=train_steps_per_epoch,
                    validation_data=val_generator,
                    validation_steps=val_steps_per_epoch,
                    callbacks=callbacks,
                    verbose=2)

    m.save(os.path.join(os.getcwd(), 'weights-final.hdf5'))


if __name__ == '__main__':
    train()

