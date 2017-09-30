from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose, BatchNormalization
from keras.models import Model

import tensorflow as tf
def leaky_relu(x, alpha=0.1):
    return tf.maximum(x, x * alpha)

def get_unet_8(img_rows, img_cols):
    inputs = Input((img_rows, img_cols, 3))
    conv1 = Conv2D(32, (3, 3), activation=leaky_relu, padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation=leaky_relu, padding='same')(conv1)
    batch1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(batch1)

    conv2 = Conv2D(64, (3, 3), activation=leaky_relu, padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation=leaky_relu, padding='same')(conv2)
    batch2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(batch2)

    conv3 = Conv2D(128, (3, 3), activation=leaky_relu, padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation=leaky_relu, padding='same')(conv3)
    batch3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(batch3)

    conv4 = Conv2D(256, (3, 3), activation=leaky_relu, padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation=leaky_relu, padding='same')(conv4)
    batch4 = BatchNormalization()(conv4)

    up5 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(batch4), batch3], axis=3)
    conv5 = Conv2D(128, (3, 3), activation=leaky_relu, padding='same')(up5)
    conv5 = Conv2D(128, (3, 3), activation=leaky_relu, padding='same')(conv5)
    batch5 = BatchNormalization()(conv5)

    up6 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(batch5), batch2], axis=3)
    conv6 = Conv2D(64, (3, 3), activation=leaky_relu, padding='same')(up6)
    conv6 = Conv2D(64, (3, 3), activation=leaky_relu, padding='same')(conv6)
    batch6 = BatchNormalization()(conv6)

    up7 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(batch6), batch1], axis=3)
    conv7 = Conv2D(32, (3, 3), activation=leaky_relu, padding='same')(up7)
    conv7 = Conv2D(32, (3, 3), activation=leaky_relu, padding='same')(conv7)
    batch7 = BatchNormalization()(conv7)

    conv8 = Conv2D(1, (1, 1), activation='sigmoid')(batch7)

    model = Model(inputs=[inputs], outputs=[conv8])

    return model