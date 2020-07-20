from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from ops import resolve_shape
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


# eraser augmentation
def do_eraser_off_graph(x, eraser_prob=0.5, boxes_max=50, boxes_min=150):
    """
    Performs eraser (cutout) augmentation on x.
    It replaces rectangular areas (boxes) in x with zero intensity pixel.
    The number of areas is sampled from a uniform distribution within a specified interval [boxes_min, boxes_max]
    This function is wrapping this functionality within a tensorflow session after reading x from the dataset record
    :param x:  (B,H,W,C)
    :param eraser_prob: propability of performing the augmentation on any given x
    :param boxes_max: max number of rectangular areas to be zero-ed out in x
    :param boxes_min: min number of rectangular areas to be zero-ed out in x
    :return: x_erased: x after the areas have been zero-ed out
    """
    do_eraser = np.random.uniform(0, 1, 1)
    # perform the augmentation with eraser_prop propability
    if do_eraser > 1.0 - eraser_prob:
        for i in range(x.shape[0]):
            # sample a random number of rectangular areas to be zero-ed out
            rand_num_boxes = np.random.randint(boxes_max, boxes_min)
            x[i, :, :, 0] = erase(x[i, :, :, 0], num_boxes=rand_num_boxes,
                                  s_l=0.0003, s_h=0.0004, r_1=1, r_2=1)
    return x


def erase(input_img, num_boxes=100, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1 / 0.3):

    """ erases a number of boxes, placed randomly within the input_img w
    # the default settings have been tuned to apply erasing to a reasonable extent.
    Adapted from https://github.com/yu4u/cutout-random-erasing
    :param input_img: the input image of shape (H, W, C) or (H,W)
    :param num_boxes: number of regions
    :param s_l: minimum fraction of erased area over input_img area
    :param s_h: maximum fraction of erased area over input_img area
    :param r_1: minimum aspect ratio of erased area
    :param r_2: maximum aspect ratio of erased area
    :return: input_img with erased areas
    """
    h, w = input_img.shape
    for box in range(num_boxes):
        while True:
            s = np.random.uniform(s_l, s_h) * h * w  # random size
            r = np.random.uniform(r_1, r_2)  # random aspect ratio
            w_box = int(np.sqrt(s / r))
            h_box = int(np.sqrt(s * r))
            col = np.random.randint(0, w)  # random col
            row = np.random.randint(0, h)  # random row
            # if staying within image boundaries proceed with next box else resample another box
            if col + w_box <= w and row + h_box <= h:
                break

        # erase the box
        input_img[row:row + h_box, col:col + w_box] = input_img.min()
    return input_img


# deformation augmentation
def do_deformation_off_graph(x_value, y_value, deform_prob=0.5,
                             alpha_max=250, alpha_min=225, sigma_max=18, sigma_min=12):
    """
    Performs deformation augmentation on (x,y).
    This function is wrapping this functionality within a tensorflow session after reading x from the dataset record
    The default arguments have been tuned for OCTA images to produce reasonable augmentations

    :param x_value: image of shape (B,H,W,C)
    :param y_value: mask of shape (B,H,W,1)
    :param deform_prob: propability of performing the augmentation on any given (x,y)
    :param alpha_max: strength of deformation parameter max
    :param alpha_min: strength of deformation parameter min
    :param sigma_max: smoothness of deformation parameter max
    :param sigma_min: smoothness of deformation parameter min
    :return: x_value and y_value deformed
    """

    for b in range(x_value.shape[0]):
        x = x_value[b, :, :, 0]
        y = y_value[b, :, :, 0]
        do_deformation = np.random.uniform(0, 1, 1)
        if do_deformation > 1 - deform_prob:
            # concatenate (x,y)
            x_y_merged = np.concatenate((x[..., None], y[..., None]), axis=2)

            # get random alpha and sigma within specified ranges
            random_alpha = np.random.uniform(alpha_min, alpha_max, 1)
            random_sigma = np.random.randint(sigma_min, sigma_max)
            x_y_deformed = elastic_deformation(x_y_merged, alpha=random_alpha, sigma=random_sigma)
            # x_y_deformed is split back to (x,y)
            x_value[b, :, :, 0] = x_y_deformed[..., 0]
            y_value[b, :, :, 0] = x_y_deformed[..., 1]
        return x_value, y_value


def elastic_deformation(image, sigma=10, alpha=100, random_state=None, warp_mode='constant'):
    """   Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
        Warps image with a random defomation field generated using samples from a spatial guassian(sigma)
    :param image: (h,w,2) if deformation is to be applied on both the ground truth and
                  the input. Else it should be (h,w).
    :param sigma: controls smoothness, something in the range 10-20
    :param alpha: controls strength something in the range 100-200 to avoid non-plausible deformations
    :param random_state: random seed for reproducibile randomness
    :param warp_mode: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
    :return: the warped image
    """

    if random_state is None:
        random_state = np.random.RandomState(None)

        shape = image.shape
        if len(shape) == 3:
            dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma)
            dx = dx * alpha

            dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma)
            dy = dy * alpha

            dz = np.zeros_like(dx)

            x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))

            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))

            return map_coordinates(image, indices, order=1, mode=warp_mode).reshape(shape)
        # order = 1 means nearest neigh

        elif len(shape) == 2:
            dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma)
            # print(dx)
            dx = dx * alpha

            dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma)
            # print(dy)
            dy = dy * alpha

            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))

            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

            return map_coordinates(image, indices, order=1, mode=warp_mode).reshape(shape)


# scaling augmentation
def random_scaling(x, y,
                   min_relative_random_scale_change=0.9, max_realtive_random_scale_change=1.1, seed=1):
    """
    perfroms random scaling on a pair of tensors (x,y)
    :param x: tensor of shape (b,h,w,c)
    :param y: tensor of shape (b,h,w,c)
    :param min_relative_random_scale_change:
    :param max_realtive_random_scale_change:
    :param seed:
    :return:
    """
    input_shape = resolve_shape(x)
    input_shape_float = tf.to_float(input_shape)
    # generating the same random value for vertical and horizontal scaling
    scaling_factor = tf.random_uniform(shape=[1],
                                       minval=min_relative_random_scale_change,
                                       maxval=max_realtive_random_scale_change,
                                       dtype=tf.float32,
                                       seed=seed)

    input_shape_scaled = tf.to_int32(tf.round(tf.multiply(input_shape_float, scaling_factor)))
    # scale x
    x_resized = tf.image.resize_images(images=x, size=tf.stack([input_shape_scaled[0], input_shape_scaled[1]], axis=0))
    # crop  or pad to go back to original image shape
    x_cropped_or_padded = tf.image.resize_image_with_crop_or_pad(x_resized, input_shape[0], input_shape[1])

    # do the two above steps for y
    y_resized = tf.image.resize_images(images=y, size=tf.stack([input_shape_scaled[0], input_shape_scaled[1]], axis=0))
    y_cropped_or_padded = tf.image.resize_image_with_crop_or_pad(y_resized, input_shape[0], input_shape[1])

    return x_cropped_or_padded, y_cropped_or_padded
