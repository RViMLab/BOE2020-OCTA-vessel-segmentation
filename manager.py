import tensorflow as tf
import numpy as np
import os
import math
import cv2
# from PIL import Image
import matplotlib.pyplot as plt

from readers import Reader
from augmentations import do_deformation_off_graph, do_eraser_off_graph
from networks import iunet, shn, unet
from utils import select_ckpt_file, save_2d_segmentation, count_records_in_tfrecord, \
    make_optimizer, correctness_completeness_quality, track_mean_value_per_epoch
from losses import sigmoid_cross_entropy_balanced, iterative_loss, perceptual_loss


class ModelManager:
    eligible_loss_types_per_model = {'UNET': ['bce', 'bce-topo'],
                                     'iUNET': ['i-bce', 'i-bce-equal', 'i-bce-topo', 'i-bce-topo-equal'],
                                     'SHN': ['s-bce', 's-bce-topo']}

    def __init__(self,
                 name=None,
                 num_layers=3,
                 feature_maps_root=64,
                 norm_type='bn',
                 n_modules=None,
                 n_iterations=None,
                 verbose=False):
        """
        :param name: name of the model used to set up the corresponding graph
        :param feature_maps_root: feature maps at root layer of any UNET module
        :param num_layers: number of layers in encoder and in decoder parts of any UNET module
        :param n_modules: number of UNET modules if using SHN else unused, defaults to None
        :param n_iterations: number of iterations if using iUNET else unused, defaults to None
        :param verbose: if True everything added to the graph will be printed else silenced graph definition
        """
        # validate arguments
        assert(name in ['UNET', 'iUNET', 'SHN'])
        if name is 'iUNET':
            assert(n_iterations is not None)
        if name is 'SHN':
            assert(n_modules is not None)

        # to be set only if a model is trained
        self.tag = None
        self.model_save_dir = None
        self.loss_type = None
        self.train_op = tf.no_op

        #
        self.vgg19_link = 'http://download.tensorflow.org/models/vgg_19_2016_08_28.tar.gz'

        # Network graph settings
        self.name = name  # name of the model used to set up the corresponding graph
        self.feature_maps_root = feature_maps_root  # feature maps at root layer of any UNET module
        self.num_layers = num_layers  # number of layers in encoder and in decoder parts of any UNET module
        self.norm = norm_type  # normalization type ('bn' is used in all pretrained networks in this repository)
        self.n_modules = n_modules  # number of UNET modules if using SHN else unused
        self.n_iterations = n_iterations  # number of iterations if using iUNET else unused
        # Network outputs in lists:
        # Explanation:
        # if using stacked (SHN) or iterative models (iUNET) then
        # the logits and sigmoided_logits are lists of the tensors corresponding to intermediate and final outputs
        # ex. logits[0], sigmoided_logits[0] contain the first iteration's logits and sigmoid outputs respectively
        # for iUNET or the first module's output for Stacked Hourglass Network (SHN)
        self.logits = []
        self.sigmoided_logits = []

        self._network_graph_def(verbose)

    def _set_tag_and_create_model_dir(self, vgg_fmaps, vgg_weights, model_dir):
        # utility method for getting a tag of each model to be used for logging and saving weights
        # uses settings of the network architecture and loss function
        self.tag = 'L_{}_F_{}_loss_{}'.format(self.num_layers, self.feature_maps_root, self.loss_type)
        if vgg_fmaps is not None and vgg_weights is not None:
            self.tag = self.tag + ''
            for f in vgg_fmaps:
                f_ = f.split('/')[-1:][0]
                f_ = 'C' + f_[-3:]
                self.tag = self.tag + '_' + f_
            for w in vgg_weights:
                self.tag = self.tag + '_' + str(w)

        if self.n_iterations is not None:
            self.tag = self.tag + '_' + str(self.n_iterations)
        elif self.n_modules is not None:
            self.tag = self.tag + '_' + str(self.n_modules)

        self.model_save_dir = os.path.join(model_dir, self.name, self.tag)
        if not os.path.exists(self.model_save_dir):
            os.makedirs(self.model_save_dir)

        print('[2]: model_save_dir was created: [{}]'.format(self.model_save_dir))

    def _network_graph_def(self, verbose=False):
        """
        defines the network graph
        :param verbose: if True prints messages as it defines layers
        :return:
        """
        # placeholders for input and ground truth
        with tf.name_scope('input'):
            self.x = tf.placeholder(name='x', shape=(None, None, None, 1), dtype=tf.float32)
            self.y = tf.placeholder(name='y', shape=(None, None, None, 1), dtype=tf.float32)
        # by default we use batch norm in train_mode (i.e with current
        if self.name == 'UNET':
            self.sigmoided_logits, self.logits, _, _ = unet(self.x, 1, self.num_layers, self.feature_maps_root,
                                                            self.norm, True, verbose=verbose)
        elif self.name == 'iUNET':
            self.sigmoided_logits, self.logits, _, _ = iunet(self.x, 1, self.num_layers, self.feature_maps_root,
                                                             self.norm, True, self.n_iterations, verbose=verbose)
        elif self.name == 'SHN':
            self.sigmoided_logits, self.logits, _, _ = shn(self.x, 1, self.num_layers, self.feature_maps_root,
                                                           self.norm, True, self.n_modules, verbose=verbose)

        self.output = self.sigmoided_logits if self.name == 'UNET' else self.sigmoided_logits[-1]
        self.variables = [v for v in tf.global_variables() if self.name in v.name]

    def _loss_def(self, loss_type, vgg_fmaps=None, vgg_weights=None):
        """
        wrapper function that sets the loss function to be used during training
        :param loss_type:
        :param vgg_fmaps: list of the vgg feature maps used for the perceptual loss
        :param vgg_weights: list of weights signifying the importance of each vgg feature map in the loss
        :return:
        """
        assert (loss_type in self.eligible_loss_types_per_model[self.name]),\
            'loss_type: [{}] not eligible for model: [{}] eligible losses for it are:[{}]'.format(loss_type, self.name, self.eligible_loss_types_per_model[self.name])

        iteration_weighing = 'equal' if 'equal' in loss_type else 'increasing'
        # iteration_weighing: if 'equal' then iUNET loss terms for each intermediate output is weighed equally else
        # uses gradually increasing weights
        print('[1]: Loss definition')
        print('loss type [{}], iteration_weighing [{}], vgg_fmaps [{}], vgg_weights [{}]'.format(loss_type,
                                                                                                 iteration_weighing,
                                                                                                 vgg_fmaps,
                                                                                                 vgg_weights))
        self.loss_type = loss_type
        with tf.name_scope('loss'):
            if loss_type == 'bce':
                # balanced binary cross entropy (not for stacked (SHN) nor for iterative models (iUNET)
                self.loss = sigmoid_cross_entropy_balanced(self.logits, self.y, name='bce')
                self.bce_raw_summary = tf.summary.scalar('bce', self.loss)
                self.loss_summaries = [self.bce_raw_summary]
                # return loss, [bce_raw_summary]
                # summary is placed in a list because it is just a tensor --> error in merge
            elif loss_type == 'bce-topo':
                # bce + topological loss (not for stacked nor for iterative models)
                self.loss_bce = sigmoid_cross_entropy_balanced(self.logits, self.y, name='bce')
                self.loss_topo = perceptual_loss(tf.nn.sigmoid(self.logits), self.y, vgg_fmaps, vgg_weights)
                self.bce_raw_summary = tf.summary.scalar('bce', self.loss_bce)
                self.topo_raw_summary = tf.summary.scalar('topo', self.loss_topo)
                self.loss = self.loss_bce + self.loss_topo
                self.loss_summaries = [self.bce_raw_summary, self.topo_raw_summary]

            elif loss_type == 'i-bce' or loss_type == 'i-bce-equal':
                # iterative loss with bce terms (only for iUNET)
                self.loss, self.loss_summaries = iterative_loss(self.sigmoided_logits, self.logits, self.y,
                                                                n_iterations=self.n_iterations,
                                                                iteration_weighing=iteration_weighing)

            if loss_type == 'i-bce-topo' or loss_type == 'i-bce-topo-equal':
                # iterative loss with bce + topological (only for iUNET)
                self.loss, self.loss_summaries = iterative_loss(self.sigmoided_logits, self.logits, self.y,
                                                                n_iterations=self.n_iterations,
                                                                iteration_weighing=iteration_weighing,
                                                                use_vgg_loss=True, vgg_fmaps=vgg_fmaps,
                                                                vgg_weights=vgg_weights)

            if loss_type == 's-bce':
                # iterative loss with bce (only for SHN)
                # each intermediate output's term is weighed equally i.e iteration_weighing='equal'
                self.loss, self.loss_summaries = iterative_loss(self.sigmoided_logits, self.logits, self.y,
                                                                n_iterations=self.n_modules,
                                                                iteration_weighing='equal')

            if loss_type == 's-bce-topo':
                # iterative loss with bce + topological (only for SHN)
                # each intermediate output's term is weighed equally i.e iteration_weighing='equal'
                self.loss, self.loss_summaries = iterative_loss(self.sigmoided_logits, self.logits, self.y,
                                                                n_iterations=self.n_modules,
                                                                iteration_weighing=iteration_weighing,
                                                                use_vgg_loss=True, vgg_fmaps=vgg_fmaps,
                                                                vgg_weights=vgg_weights)

    def train(self, train_tfrecord, loss_type, vgg_fmaps=None, vgg_weights=None, validation_tfrecord=None,
              training_steps=6000, batch_size=2,
              initial_lr=10**(-4), decay_steps=2000, decay_rate=0.5, do_online_augmentation=True,
              log_dir='', model_dir='models'):
        """
        method used to train a network, specify a loss, logging and saving the trained model
        :param train_tfrecord: a tfrecord for training data (see readers.py for details on the expected format)
        :param loss_type: a string that can be
                     'UNET': ['bce', 'bce-topo']
                     'iUNET': ['i-bce', 'i-bce-equal', 'i-bce-topo', 'i-bce-topo-equal']
                     'SHN': ['s-bce', 's-bce-topo']

        :param vgg_fmaps: a list of the names of the vgg feature maps to be used for the perceptual loss
        :param vgg_weights: a list of weights, each controlling the importance of a feature map of the
                            perceptual loss in the total loss
        :param validation_tfrecord: (optional) a tfrecord to keep track of during training
        :param training_steps: total number of training steps
        :param batch_size: batch size for the optimizer
        :param initial_lr: starting learning rate
        :param decay_steps: learning rate decay steps
        :param decay_rate: rate of decay of the lr (see make_optimizer, make_learning_rate_scheduler utils.py)
        :param do_online_augmentation: if True performs online data augmentation using
                                       the default settings (see readers.py)
        :param log_dir: path to dir where the logs will be stored
        :param model_dir: path to dir where the models will be stored
        :return:
        """

        if 'topo' in loss_type and vgg_weights is None and vgg_fmaps is None:
            # default settings for the topological loss function
            vgg_fmaps = ['vgg_19/conv1/conv1_2', 'vgg_19/conv2/conv2_2', 'vgg_19/conv3/conv3_4']
            vgg_weights = [0.01, 0.001, 0.0001]

        train_examples = count_records_in_tfrecord(train_tfrecord)
        epoch_step = math.floor(train_examples / batch_size)

        # define the loss: sets self.loss and self.loss_summaries
        self._loss_def(loss_type=loss_type, vgg_weights=vgg_weights, vgg_fmaps=vgg_fmaps)

        # generate a tag for logging purposes and saving directory naming
        self._set_tag_and_create_model_dir(vgg_fmaps, vgg_weights, model_dir)

        # using the default optimizer settings
        with tf.name_scope('optimizer'):
            self.train_op, learning_rate_summary, grads_and_vars_summary = make_optimizer(self.loss, self.variables,
                                                                                          lr_start=initial_lr,
                                                                                          lr_scheduler_type='inverse_time',
                                                                                          decay_steps=decay_steps,
                                                                                          decay_rate=decay_rate,
                                                                                          name='Adam_optimizer')

        # summaries
        input_summary = tf.summary.image('input MIP ', self.x, max_outputs=1)
        ground_truth_summary = tf.summary.image('ground truth', self.y, max_outputs=1)
        output_summary = tf.summary.image('output', self.output, max_outputs=1)

        train_summary = tf.summary.merge([learning_rate_summary, output_summary]+self.loss_summaries)
        valid_summary = tf.summary.merge([input_summary, output_summary, ground_truth_summary])

        # readers and saver
        train_reader = Reader(train_tfrecord, image_size=416, channels=1, batch_size=batch_size,
                              do_online_augmentation=do_online_augmentation, do_shuffle=True, name='train_reader')
        x_train, y_train = train_reader.feed()

        # if there is a validation set
        validation_step, validation_examples = -1, 0  # just to supress warnings
        if validation_tfrecord is not None:
            validation_examples, validation_step, best_quality = count_records_in_tfrecord(validation_tfrecord), epoch_step, 0
            validation_reader = Reader(validation_tfrecord, image_size=416, channels=1, batch_size=1, do_shuffle=False,
                                       do_online_augmentation=False, name='validation_reader')
            x_test, y_test = validation_reader.feed()

        saver = tf.train.Saver(max_to_keep=1, var_list=self.variables)

        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'logs', self.name, self.tag), sess.graph)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            sess.run(tf.global_variables_initializer())

            # restore vgg19 for the topological loss term
            # requires pretrained vgg_19 to be inside the path
            if 'topo' in loss_type:
                assert(os.path.isfile('vgg_19.ckpt')), 'vgg_19.ckpt must be in the path for ' \
                                                       'training with loss_type=[{}] please ' \
                                                       'download from {}'.format(self.loss_type, self.vgg19_link)

                vgg_restore_list = [v for v in tf.global_variables() if 'vgg_19' in v.name and 'Adam' not in v.name]
                restorer_vgg = tf.train.Saver(var_list=vgg_restore_list)
                restorer_vgg.restore(sess, 'vgg_19.ckpt')

            epoch, costs, losses_train = 0, [], []
            for i in range(training_steps):
                x_value, y_value = sess.run([x_train, y_train])

                # below the two augmentation functions are implemented outside tensorflow
                if do_online_augmentation:
                    x_value, y_value = do_deformation_off_graph(x_value, y_value, deform_prob=0.5)
                    x_value = do_eraser_off_graph(x_value, None, eraser_prob=0.3,  boxes_max=50, boxes_min=150)

                train_feed = {self.x: x_value, self.y: y_value}
                loss_batch, summary_train, _ = sess.run([self.loss, train_summary, self.train_op], feed_dict=train_feed)
                train_writer.add_summary(summary_train, i)

                epoch, losses_train, loss_mean, train_writer = track_mean_value_per_epoch(i, loss_batch, epoch_step,
                                                                                          epoch, losses_train,
                                                                                          train_writer,
                                                                                          tag='loss-per-epoch')
                # validation
                if ((i % validation_step == 0) or (i == training_steps-1)) and i > 0 and validation_tfrecord is not None:
                    sigmoided_out_list = []
                    y_test_value_list = []
                    for test_step in range(validation_examples):
                        x_test_value, y_test_value = sess.run([x_test, y_test])
                        test_feed = {self.x: x_test_value, self.y: y_test_value}
                        sigmoided_out, summary_val = sess.run([self.output, valid_summary], feed_dict=test_feed)
                        train_writer.add_summary(summary_val, i+test_step)
                        sigmoided_out_list.append(sigmoided_out)
                        y_test_value_list.append(y_test_value)

                    # run metric (computes means across the validation set)
                    correctness, completeness, quality, _ = correctness_completeness_quality(y_test_value_list,
                                                                                             sigmoided_out_list,
                                                                                             threshold=0.5)
                    new_quality = quality
                    if best_quality < new_quality:
                        diff = new_quality - best_quality
                        print('EPOCH:', epoch,
                              'completness:', completeness,
                              'correctness:', correctness,
                              'quality:', quality,
                              'previous quality:', best_quality,
                              'NEW_BEST with difference:', diff)
                        best_quality = new_quality
                        save_path = saver.save(sess, model_dir + '/' + self.name + '_' + str(i) + ".ckpt")
                        print("Model saved in path: %s" % save_path)
                    else:
                        print('EPOCH:', epoch,
                              'completness:', completeness, 'correctness:', correctness, 'quality:', quality)
                    summary_metrics = tf.Summary()
                    summary_metrics.value.add(tag='completness', simple_value=completeness)
                    summary_metrics.value.add(tag='correctness', simple_value=correctness)
                    summary_metrics.value.add(tag='quality', simple_value=quality)
                    train_writer.add_summary(summary_metrics, i)

                if i == (training_steps - 1):
                    save_path = saver.save(sess, os.path.join(self.model_save_dir + self.name + '_' + str(i) + ".ckpt"))
                    print("Final Model saved in path: %s" % save_path)

            coord.request_stop()
            coord.join(threads)
        train_writer.close()

    def run_on_images(self, path_to_dir, path_to_model_ckpt_dir, get_intermediate_outputs=False, show_outputs=False, path_to_save_dir=None):
        """
        performs segmentation on all images within a directory
        :param path_to_dir: path to directory of images to be segmented by the pre-trained model
        :param path_to_model_ckpt_dir: path to directory where the pre-trained network is stored
        :param get_intermediate_outputs: if True plots and saves intermediate outputs for iUNET and SHN
                                         (must be False if model name is UNET)
                if saved the intermediate outputs for each image follow the naming convention <image_name>_<model_name>_<number_of_intermediate_output>

        :param show_outputs: if True shows the segmented images
        :param path_to_save_dir: if None then segmentations are not save else must be the path to the save dir which is created if it does not exist

        :return:
        """
        if self.name == 'UNET':
            assert(not get_intermediate_outputs)
        if path_to_save_dir is not None:
            if not os.path.exists(path_to_save_dir):
                os.makedirs(path_to_save_dir)
        path_to_ckpt_file = select_ckpt_file(model_dir=path_to_model_ckpt_dir)  # utility to get the right file
        saver = tf.train.Saver(var_list=self.variables)
        paths_to_files = [os.path.join(path_to_dir, f) for f in os.listdir(path_to_dir)]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            # restore checkpoint file
            saver.restore(sess, path_to_ckpt_file)
            print('restored weights from', path_to_ckpt_file)
            for path_to_file in paths_to_files:
                # img = np.array(Image.open(path_to_file), dtype=np.uint8) / 255
                img = cv2.imread(path_to_file) / 255
                print('{}: img.shape: {}'.format(path_to_file, img.shape))
                if not img.shape == (416, 416):
                    # this is not necessary but best results are obtained for this resolution
                    img = cv2.resize(img, dsize=(416, 416), interpolation=cv2.INTER_LINEAR)
                    # in case images have 3 channels
                    img = img[..., 0] if img.shape == (416, 416, 3) else img
                # making the image a 4d tensor
                x_test_value = img[np.newaxis, :, :, np.newaxis]
                test_feed = {self.x: x_test_value}

                if get_intermediate_outputs:
                    n = self.n_modules if self.name == 'SHN' else self.n_iterations
                    ys = [sess.run(self.sigmoided_logits[i], feed_dict=test_feed) for i in range(n)]

                    if show_outputs:
                        for y in ys:
                            plt.imshow(y[0, :, :, 0] > 0.5, cmap='hot')
                            plt.imshow(x_test_value[0, :, :, 0], alpha=0.6, cmap='gray')
                            plt.show(block=False)
                            plt.pause(0.25)

                    if path_to_save_dir is not None:
                        r = 'iteration' if self.name == 'iUNET' else 'module'
                        for y, i in zip(ys, range(n)):
                            filename = path_to_file.split('\\')[-1][:-4]
                            save_2d_segmentation(y, '{}_{}_{}_{}.png'.format(filename, self.name, r, i+1), path_to_save_dir)
                else:
                    y = sess.run(self.output, feed_dict=test_feed)

                    if show_outputs:
                        plt.imshow(y[0, :, :, 0], cmap='gray')
                        plt.show(block=False)
                    if path_to_save_dir is not None:
                        filename = path_to_file.split('\\')[-1][:-4]
                        save_2d_segmentation(y, 'img:{}_{}.png'.format(filename, self.name), path_to_save_dir)
