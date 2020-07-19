import tensorflow as tf
from augmentations import random_scaling
from utils import identity_on_tensors
from collections import OrderedDict


class Reader:

    def __init__(self, tf_records_file, image_size=416, channels=1, do_online_augmentation=False,
                 augmentation_details=None, do_shuffle=True, read_pathology=False,
                 min_queue_examples=10, batch_size=1, num_threads=8, name='Reader'):

        """
        Args:
          tf_records_file: string, tfrecords file path
          min_queue_examples: integer, minimum number of samples to retain in the queue that provides of batches of examples
          batch_size: integer, number of images per batch
          num_threads: integer, number of preprocess threads
        """

        self.tfrecords_file = tf_records_file
        self.image_size = image_size  # assumes w=h
        self.channels = channels
        self.min_queue_examples = min_queue_examples
        self.batch_size = batch_size
        self.num_threads = num_threads
        self.reader = tf.TFRecordReader()
        self.name = name
        # augmentations and shuffling
        self.do_shuffle = do_shuffle
        self.do_online_augmentation = do_online_augmentation
        self.read_pathology = read_pathology
        self.augmentation_details = augmentation_details

        # augmentation settings for augmentations handled within tensorflow
        # deformation and cutout augmentations are handled externally using numpy and scipy functions
        # (see train() in manager.py )
        if augmentation_details is not None:
            self.augmentation_details = augmentation_details
        else:
            self.augmentation_details = self._get_default_augmentation_details()

        if self.augmentation_details['contrast_distortion_lower'] < 0.2:
            raise ValueError('contrast_distortion_lower [{}]'
                             ' is too low'.format(self.augmentation_details['contrast_distortion_lower']))

    @staticmethod
    def _parse_tfrecord_features(serialized_example):
        features = tf.parse_single_example(serialized_example,
                                           features={'pathology': tf.FixedLenFeature([], tf.int64),
                                                     'height': tf.FixedLenFeature([], tf.int64),
                                                     'width': tf.FixedLenFeature([], tf.int64),
                                                     'channels': tf.FixedLenFeature([], tf.int64),
                                                     'image_raw': tf.FixedLenFeature([], tf.string),
                                                     'mask_raw': tf.FixedLenFeature([], tf.string)})
        return features

    @staticmethod
    def _get_default_augmentation_details():
        augmentation_details = OrderedDict()

        augmentation_details['scaling_prob'] = 0.5
        augmentation_details['scaling_max'] = 1.3
        augmentation_details['scaling_min'] = 0.8

        augmentation_details['brightness_distortion_prob'] = 0.5
        augmentation_details['brightness_distortion_delta'] = 0.2

        augmentation_details['contrast_distortion_prob'] = 0.5
        augmentation_details['contrast_distortion_upper'] = 1.25
        augmentation_details['contrast_distortion_lower'] = 0.75

        return augmentation_details

    def _augmentation(self, image, mask, seed=1):
        # receives an image and a mask and applies with some set propability intensity and geometry transformations
        # currently supports brightness, contrast and scaling
        with tf.name_scope('brightness_distortion'):
            do_brightness = tf.random_uniform([1, 1], minval=0, maxval=1, dtype=tf.float32, seed=seed)
            image = tf.cond(do_brightness[0][0] > 1-self.augmentation_details['brightness_distortion_prob'],
                            lambda: tf.image.random_brightness(image, max_delta=self.augmentation_details['brightness_distortion_delta'],
                                                               seed=seed),
                            lambda: image)
            image = tf.clip_by_value(image, 0, 1)
        with tf.name_scope('contast_distortion'):
            do_contrast = tf.random_uniform([1, 1], minval=0, maxval=1, dtype=tf.float32, seed=seed)
            image = tf.cond(do_contrast[0][0] > 1 - self.augmentation_details['contrast_distortion_prob'],
                            lambda: tf.image.random_contrast(image,
                                                             upper=self.augmentation_details['contrast_distortion_upper'],
                                                             lower=self.augmentation_details['contrast_distortion_lower'],
                                                             seed=seed),
                            lambda: image)
            image = tf.clip_by_value(image, 0, 1)

        with tf.name_scope('random_scaling'):
            do_scaling = tf.random_uniform([1, 1], minval=0, maxval=1, dtype=tf.float32, seed=seed)
            image, mask = tf.cond(do_scaling[0][0] > 1-self.augmentation_details['scaling_prob'],
                                  lambda: random_scaling(image, mask, min_relative_random_scale_change=self.augmentation_details['scaling_min'],
                                                         max_realtive_random_scale_change=self.augmentation_details['scaling_max'], seed=seed),
                                  lambda: identity_on_tensors(image, mask))

        return image, mask

    def feed(self):
        """
        Returns:
          images: 4D tensor [batch_size, image_width, image_height, channels]
        """
        with tf.name_scope(self.name):
            filename_queue = tf.train.string_input_producer([self.tfrecords_file])
            reader = tf.TFRecordReader()
            _, serialized_example = reader.read(filename_queue)
            features = self._parse_tfrecord_features(serialized_example)

            # read image dimensions
            height = tf.cast(features['height'], tf.int32)
            width = tf.cast(features['width'], tf.int32)
            channels = tf.cast(features['channels'], tf.int32)

            # read pathology
            p = tf.cast(features['pathology'], tf.int32)
            p_batch = tf.no_op('no_op_for_pathology')

            # read encoded image/mask and reshape and cast them to appropriate format
            image_buffer = features['image_raw']
            mask_buffer = features['mask_raw']

            image = tf.decode_raw(image_buffer, tf.uint8)
            image = tf.reshape(image, tf.stack([self.image_size, self.image_size, 1]))

            mask = tf.decode_raw(mask_buffer, tf.uint8)
            mask = tf.reshape(mask, tf.stack([self.image_size, self.image_size, 1]))

            image = tf.cast(image, dtype=tf.float32) / 255
            mask = tf.cast(mask, dtype=tf.float32) / 255

            if self.do_online_augmentation:
                image, mask = self._augmentation(image, mask)

            if self.do_shuffle:  # when training
                image_batch, mask_batch = tf.train.shuffle_batch([image, mask],
                                                                 shapes=[[self.image_size, self.image_size, self.channels],
                                                                         [self.image_size, self.image_size, 1]],
                                                                 batch_size=self.batch_size,
                                                                 num_threads=self.num_threads,
                                                                 capacity=self.min_queue_examples + 3*self.batch_size,
                                                                 min_after_dequeue=self.min_queue_examples)
            else:  # when testing
                image_batch, mask_batch = tf.train.batch([image, mask],
                                                         shapes=[[self.image_size, self.image_size, self.channels], [self.image_size, self.image_size, 1]],
                                                         batch_size=self.batch_size,
                                                         num_threads=self.num_threads,
                                                         capacity=self.min_queue_examples + 3 * self.batch_size)

            return image_batch, mask_batch
