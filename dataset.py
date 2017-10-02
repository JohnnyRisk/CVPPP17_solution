
from scipy import misc
import numpy as np
from glob import glob
import os
from keras.preprocessing.image import ImageDataGenerator
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def load_images(data_dir, sub_dir='train', img_size=(256, 256), color_channel=3, mask_channel=1):
    # get the images and the masks path to search for
    image_path = os.path.join(data_dir, sub_dir, 'images', '*.png')
    mask_path = os.path.join(data_dir, sub_dir, 'masks', '*.png')

    # now we grab the files from the folder
    image_files = glob(image_path)
    mask_files = glob(mask_path)

    image_files = [img for img in sorted(image_files)]
    mask_files = [msk for msk in sorted(mask_files)]

    # make sure the files are not empty
    # TODO need to make sure that you have all the same base files so all the files have labels
    if len(image_files) == 0:
        raise Exception("No image files found in {}".format(image_path))

    if len(mask_files) == 0:
        raise Exception("No image files found in {}".format(mask_path))

    # we go grab the files and create a list of images resized (loads everything in memory)
    # need to change this to make it scalable
    images = [misc.imresize(misc.imread(i), (img_size[0], img_size[1], color_channel)) for i in image_files]
    masks = [misc.imresize(misc.imread(i), (img_size[0], img_size[1], mask_channel), interp='nearest') / 255 for i in
             mask_files]

    # convert the lists to arrays
    images = np.asarray(images)
    masks = np.asarray(masks)

    # the color channels get squashed out if they are 1. this is one way to fix that.
    if color_channel == 1:
        images = np.expand_dims(images, axis=-1)

    if mask_channel == 1:
        masks = np.expand_dims(masks, axis=-1)

    return images, masks


from math import ceil


def random_elastic_deformation(image, alpha, sigma, mode='nearest',
                               random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape) == 3

    if random_state is None:
        random_state = np.random.RandomState(None)

    height, width, channels = image.shape

    dx = gaussian_filter(2 * random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter(2 * random_state.rand(height, width) - 1,
                         sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    indices = (np.repeat(np.ravel(x + dx), channels),
               np.repeat(np.ravel(y + dy), channels),
               np.tile(np.arange(channels), height * width))

    values = map_coordinates(image, indices, order=1, mode=mode)

    return values.reshape((height, width, channels))


class Iterator(object):
    def __init__(self, images, masks, batch_size,
                 shuffle=True,
                 rotation_range=180,
                 width_shift_range=0.1,
                 height_shift_range=0.1,
                 shear_range=0.1,
                 zoom_range=0.05,
                 fill_mode='nearest',
                 alpha=0,
                 sigma=0,
                 horizontal_flip=0,
                 vertical_flip=0):
        self.images = images
        self.masks = masks
        self.batch_size = batch_size
        self.shuffle = shuffle
        augment_options = {
            'rotation_range': rotation_range,
            'width_shift_range': width_shift_range,
            'height_shift_range': height_shift_range,
            'shear_range': shear_range,
            'zoom_range': zoom_range,
            'fill_mode': fill_mode,
            'horizontal_flip': horizontal_flip,
            'vertical_flip': vertical_flip
        }
        self.idg = ImageDataGenerator(**augment_options)
        self.alpha = alpha
        self.sigma = sigma
        self.fill_mode = fill_mode
        self.i = 0
        self.index = np.arange(len(images))
        if shuffle:
            np.random.shuffle(self.index)

    def __next__(self):
        return self.next()

    def next(self):
        # compute how many images to output in this batch
        start = self.i
        end = min(start + self.batch_size, len(self.images))

        augmented_images = []
        augmented_masks = []
        for n in self.index[start:end]:
            image = self.images[n]
            mask = self.masks[n]

            _, _, channels = image.shape

            # stack image + mask together to simultaneously augment
            stacked = np.concatenate((image, mask), axis=2)

            # apply simple affine transforms first using Keras
            augmented = self.idg.random_transform(stacked)

            # maybe apply elastic deformation
            if self.alpha != 0 and self.sigma != 0:
                augmented = random_elastic_deformation(
                    augmented, self.alpha, self.sigma, self.fill_mode)

            # split image and mask back apart
            augmented_image = augmented[:, :, :channels]
            augmented_images.append(augmented_image)
            augmented_mask = np.round(augmented[:, :, channels:])
            augmented_masks.append(augmented_mask)

        self.i += self.batch_size
        if self.i >= len(self.images):
            self.i = 0
            if self.shuffle:
                np.random.shuffle(self.index)

        return np.asarray(augmented_images), np.asarray(augmented_masks)


def normalize(x, epsilon=1e-7, axis=(1, 2)):
    x -= np.mean(x, axis=axis, keepdims=True)
    x /= np.std(x, axis=axis, keepdims=True) + epsilon


def create_generators(data_dir, sub_dir, batch_size, validation_split=0.0,
                      shuffle_train_val=True, shuffle=True, seed=None,
                      normalize_images=True, augment_training=False,
                      augment_validation=False, augmentation_args={}):
    images, masks = load_images(data_dir, sub_dir)

    # before: type(masks) = uint8 and type(images) = uint16
    # convert images to double-precision
    images = images.astype('float64')

    # maybe normalize image
    if normalize_images:
        normalize(images, axis=(1, 2))

    if seed is not None:
        np.random.seed(seed)

    if shuffle_train_val:
        # shuffle images and masks in parallel
        rng_state = np.random.get_state()
        np.random.shuffle(images)
        np.random.set_state(rng_state)
        np.random.shuffle(masks)

    # split out last %(validation_split) of images as validation set
    split_index = int((1 - validation_split) * len(images))

    if augment_training:
        train_generator = Iterator(
            images[:split_index], masks[:split_index],
            batch_size, shuffle=shuffle, **augmentation_args)
    else:
        idg = ImageDataGenerator()
        train_generator = idg.flow(images[:split_index], masks[:split_index],
                                   batch_size=batch_size, shuffle=shuffle)

    train_steps_per_epoch = ceil(split_index / batch_size)

    if validation_split > 0.0:
        if augment_validation:
            val_generator = Iterator(
                images[split_index:], masks[split_index:],
                batch_size, shuffle=shuffle, **augmentation_args)
        else:
            idg = ImageDataGenerator()
            val_generator = idg.flow(images[split_index:], masks[split_index:],
                                     batch_size=batch_size, shuffle=shuffle)
    else:
        val_generator = None

    val_steps_per_epoch = ceil((len(images) - split_index) / batch_size)

    return (train_generator, train_steps_per_epoch,
            val_generator, val_steps_per_epoch)
