from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from ops import resolve_shape
import numpy as np
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def do_eraser_off_graph(x, y=None, eraser_prob=0.5, boxes_max=50, boxes_min=150):

    do_eraser = np.random.uniform(0, 1, 1)
    if do_eraser > 1.0 - eraser_prob:
        for i in range(x.shape[0]):
            if y is None:
                x_erased = eraser_session(x[i, :, :, 0], propability=1.0, boxes_max=boxes_max, boxes_min=boxes_min)
                x[i, :, :, 0] = x_erased
            if y is not None:
                x_erased = eraser_session(x[i, :, :, 0], y=y[i, :, :, 0], propability=1.0, boxes_max=boxes_max, boxes_min=boxes_min)
                x[i, :, :, 0] = x_erased
    return x


def eraser_session(x, propability=0.5, boxes_max=200, boxes_min=100, y=None ):
    """
    Performs eraser augmentation on X
    TO be added within a session after inputs X,Y are read
    :param x:  (B,h,w,C)
    :param propability: how often to perfrom it
    :param boxes_max:
    :param boxes_min:
    :return: x_erased
    """
    rand_num_boxes = np.random.randint(boxes_max, boxes_min)
    eraser = get_random_eraser(p=propability, num_boxes=rand_num_boxes, s_l=0.0003, s_h=0.0004,
                               v_l=0, v_h=1, r_1=1, r_2=1)
    if y is None:
        x_erased = eraser(x)
        return x_erased
    else:
        x_erased = eraser(x, y)
        return x_erased


def get_random_eraser(p=0.5, num_boxes=100, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1 / 0.3, v_l=0, v_h=255,
                      pixel_level=False):
    """
    :param p: propability of performing erasing
    :param num_boxes: number of regions
    :param s_l: minimum proportion of erased area against input image
    :param s_h: maximum proportion of erased area against input image
    :param r_1: minimum aspect ratio of erased area
    :param r_2: maximum aspect ratio of erased area
    :param v_l: minimum value for erased area
    :param v_h: maximum value for erased area
    :param pixel_level:
    :return:
    """

    def eraser(input_img, mask_img=None):
        img_h, img_w = input_img.shape
        p_1 = np.random.rand()

        if p_1 > p:
            return input_img

        for box in range(num_boxes):

            while True:
                s = np.random.uniform(s_l, s_h) * img_h * img_w
                r = np.random.uniform(r_1, r_2)
                w = int(np.sqrt(s / r))
                h = int(np.sqrt(s * r))
                left = np.random.randint(0, img_w)
                top = np.random.randint(0, img_h)

                if left + w <= img_w and top + h <= img_h:
                    break

            if mask_img is not None:
                input_img[top:top + h, left:left + w] = (1 - (0.7 + np.random.uniform(-0.1, 0.1)) * mask_img[top:top + h, left:left + w]) * input_img[top:top + h, left:left + w] + \
                                                        (1 - mask_img[top:top + h, left:left + w]) * input_img[top:top + h, left:left + w]

            else:
                input_img[top:top + h, left:left + w] = input_img.min()

        return input_img

    return eraser


# def do_off_graph_augmentations(x_value, y_value, deform_prob=0.5, eraser_prob=0.5):
#     x = x_value[0, :, :, 0]
#     y = y_value[0, :, :, 0]
#     do_deformation = np.random.uniform(0, 1, 1)
#     if do_deformation > deform_prob:
#         images_merged_deformed = elastic_deformation_session(x, y,
#                                                              alpha_range=[300, 450],
#                                                              sigma_range=[12, 20])
#
#         x_value[0, :, :, 0] = images_merged_deformed[..., 0]
#         y_value[0, :, :, 0] = images_merged_deformed[..., 1]
#
#     X = x_value[0, :, :, 0]
#     do_eraser = np.random.uniform(0, 1, 1)
#     if do_eraser > eraser_prob:
#         x_erased = eraser_session(X, propability=1.0, num_boxes_range=[100, 500])
#         x_value[0, :, :, 0] = x_erased
#
#     return x_value, y_value


def do_deformation_off_graph(x_value, y_value, deform_prob=0.5,
                             alpha_max=250, alpha_min=225, sigma_max=18, sigma_min=12):

    for b in range(x_value.shape[0]):
        x = x_value[b, :, :, 0]
        y = y_value[b, :, :, 0]
        do_deformation = np.random.uniform(0, 1, 1)
        if do_deformation > 1 - deform_prob:
            images_merged_deformed = elastic_deformation_session(x, y,
                                                                 alpha_range=[alpha_min, alpha_max],
                                                                 sigma_range=[sigma_min, sigma_max])

            x_value[b, :, :, 0] = images_merged_deformed[..., 0]
            y_value[b, :, :, 0] = images_merged_deformed[..., 1]
        return x_value, y_value


def elastic_deformation(image, sigma=15, alpha=100, random_state=None, warp_mode='constant'):
    """   Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
        Warps image with a random defomation field generated using samples from a spatial guassian(sigma)
    :param image: (h,w,2) if deformation is to be applied on both the ground truth and
                  the input. Else it should be (h,w).
    :param sigma: something in the range 10-20
    :param alpha: something in the range 100-200 to avoid non-plausible deformations
    :param random_state: random seed for reproducibile randomness
    :param warp_mode: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.map_coordinates.html
                      for fundus images with a black circle around them a warp_mode = 'warp' is best!
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

            x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))  # , np.arange(shape[2]))

            indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

            return map_coordinates(image, indices, order=1, mode=warp_mode).reshape(shape)


def elastic_deformation_session(X, Y, alpha_range, sigma_range):
    """
    Performs elastic deformation on pairs of X and Y with the deformation field being
    generated using a random alpha and sigma within the specified ranges
    :param X: image (h,w)
    :param Y: annotation (h,w)
    :param alpha_range: strength of deformation
    :param sigma_range: smoothness of deformation (set higher for greated smoothness)
    :return:
    """
    im_merge = np.concatenate((X[..., None], Y[..., None]), axis=2)
    random_alpha = np.random.uniform(alpha_range[0], alpha_range[1], 1)
    random_sigma = np.random.randint(sigma_range[0], sigma_range[1])
    images_merged_deformed = elastic_deformation(im_merge,
                                                 alpha=random_alpha,
                                                 sigma=random_sigma)
    return images_merged_deformed


def random_scaling(img_tensor, annotations_tensor, min_relative_random_scale_change=0.9, max_realtive_random_scale_change=1.1, seed=1):

    input_shape = resolve_shape(img_tensor)
    # annotations_shape = resolve_shape(annotations_tensor)

    input_shape_float = tf.to_float(input_shape)

    # generating the same random value for vertical and horizontal scaling
    scaling_factor = tf.random_uniform(shape=[1],
                                       minval=min_relative_random_scale_change,
                                       maxval=max_realtive_random_scale_change,
                                       dtype=tf.float32,
                                       seed=seed)

    input_shape_scaled = tf.to_int32(tf.round(tf.multiply(input_shape_float, scaling_factor)))

    img_resized = tf.image.resize_images(images=img_tensor, size=tf.stack([input_shape_scaled[0], input_shape_scaled[1]], axis=0))
    # crop or pad to go back to original image shape
    img_cropped_or_padded = tf.image.resize_image_with_crop_or_pad(img_resized, input_shape[0], input_shape[1])

    annotations_tensor = tf.image.resize_images(images=annotations_tensor, size=tf.stack([input_shape_scaled[0], input_shape_scaled[1]], axis=0))
    # crop or pad to go back to annotations shape
    annotations_tensor = tf.image.resize_image_with_crop_or_pad(annotations_tensor, input_shape[0], input_shape[1])

    return img_cropped_or_padded, annotations_tensor
