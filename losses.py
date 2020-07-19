import tensorflow as tf
import numpy as np
import vgg_slim as vgg


def sigmoid_cross_entropy_balanced(logits, ground_truth, name='bce'):
    """
    Implements:
    balanced binary cross entropy for an output with sigmoid non-linear activation
    Based on Equation [2] from https://arxiv.org/pdf/1504.06375.pdf
    Counts foreground/background pixels for each sample and balances loss terms accordingly
    :param logits: logits
    :param ground_truth: binary ground truth
    :param name: name of loss operation
    :return: bce loss
    """

    batch_size = tf.shape(logits)[0]
    n_heatmaps = tf.shape(logits)[3]
    h = tf.shape(logits)[1]  # gets the height and width of the logits and ground truth in an online fashion
    w = tf.shape(logits)[2]

    ground_truth = tf.reshape(tf.transpose(ground_truth, [0, 3, 1, 2]), [batch_size * n_heatmaps, h * w])
    logits = tf.reshape(tf.transpose(logits, [0, 3, 1, 2]), [batch_size * n_heatmaps, h * w])

    y = tf.cast(ground_truth, tf.float32)

    count_neg = tf.reduce_sum(1. - y)
    count_pos = tf.reduce_sum(y)

    # Equation [2]
    beta = count_neg / (count_neg + count_pos)

    # Equation [2] divide by 1 - beta
    pos_weight = beta / (1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)

    # Multiply by 1 - beta
    cost = tf.reduce_mean(cost * (1 - beta))

    # check if image has no edge pixels return 0 else return complete error function
    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost, name=name)


class VggEncoder:
    def __init__(self, name='VGGEnc', vgg_id='19', cutoff='vgg_19/conv4/conv4_1'):
        self.name = name
        self.vgg_id = vgg_id
        self.reuse = False
        self.cutoff = cutoff
    def __call__(self, x, purpose, requested_vgg_feature_maps=None, multiply_by255=True, verbose=False):
        """
        *255, mean subtraction, and GRAY->RGB->BGR are done internally!!
        :param x: it expects an RGB image in [0,1]
        :param purpose:
        :param requested_vgg_feature_maps:
        :param multiply_by255: if True multiplies input by 255 to bring it to [0,255]
        :param verbose:
        :return:
        """
        if verbose:
            print('####__call__ {} ####'.format(self.name))
            raise Warning('   mean subtraction and RGB->BGR are done internally')

        with tf.name_scope(self.name):
            if x.shape[3] == 1:
                print('[VggEncoder:{}]:input is grayscale and is converted'
                      ' to rgb through channel repetition'.format(self.name))
                x = tf.image.grayscale_to_rgb(x, name='gray2rgb')

            with tf.name_scope('mean_sub_and_switch_to_BGR'):
                if multiply_by255:
                    x = x * 255
                x = VggEncoder.preprocess(x, mode='RGB')
                # x = tf.reverse(x, axis=[-1])u

            if self.vgg_id == '19':
                _, vgg_end_points = vgg.vgg_19(x, is_training=False, fc_conv_padding='SAME', global_pool=True)
            elif self.vgg_id == '16':
                _, vgg_end_points = vgg.vgg_16(x, is_training=False, fc_conv_padding='SAME', global_pool=True)
            else:
                raise ValueError('vgg_[{}] is not recognized'.format(self.vgg_id))

        # sanity checks
        if self.cutoff not in vgg_end_points.keys() and purpose in ['encoding', 'both']:
            raise ValueError('the cuttoff layer {} of the pre-trained network is not recognized'.format(self.cutoff))

        if requested_vgg_feature_maps is None and purpose == 'loss':
            raise ValueError('the vgg_feature_maps {} of the pre-trained network are not recognized'.format(
                requested_vgg_feature_maps))

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='vgg_19')

        if purpose == 'encoding':
            return vgg_end_points[self.cutoff]

        elif purpose == 'loss':
            vgg_loss_feat_maps = []
            for f in requested_vgg_feature_maps:
                vgg_loss_feat_maps.append(vgg_end_points[f])
            return vgg_loss_feat_maps

        elif purpose == 'driu':
            vgg_loss_feat_maps = []
            for f in requested_vgg_feature_maps:
                vgg_loss_feat_maps.append(vgg_end_points['DRIU/' + f])
            return vgg_loss_feat_maps

        elif purpose == 'both':
            return vgg_end_points[self.cutoff], vgg_end_points

        else:
            raise ValueError('purpoose {} of calling {} is not recoginized'.format(purpose, self.name))

    @classmethod
    def preprocess(cls, image, mode='RGB'):
        if mode == 'BGR':
            return image - np.array([103.939, 116.779, 123.68])
        else:
            return image - np.array([123.68, 116.779, 103.939])

    @classmethod
    def deprocess(cls, image, mode='RGB'):
        if mode == 'BGR':
            return image + np.array([103.939, 116.779, 123.68])
        else:
            return image + np.array([123.68, 116.779, 103.939])


def perceptual_loss(sigmoided_logits, ground_truth, vgg_fmaps, vgg_weights):
    vgg_enc = VggEncoder(name='VGG19_Encoder', cutoff='vgg_19/conv4/conv4_1')
    pred_fmaps = vgg_enc(sigmoided_logits,
                         purpose='loss',
                         requested_vgg_feature_maps=vgg_fmaps,
                         verbose=False)
    annots_fmaps = vgg_enc(ground_truth,
                           purpose='loss',
                           requested_vgg_feature_maps=vgg_fmaps,
                           verbose=False)

    loss_topo = tf.constant(0, dtype=tf.float32)
    for fm_pred, fm_gt, w in zip(pred_fmaps, annots_fmaps, vgg_weights):
        print('perceptual loss fmaps: {}--{}'.format(fm_pred, fm_gt))
        loss_topo = loss_topo + w * tf.reduce_mean((fm_pred - fm_gt) ** 2)

    return loss_topo


def iterative_loss(sigmoided_logits, logits, ground_truth, n_iterations, iteration_weighing,
                   use_vgg_loss=False, vgg_fmaps=None, vgg_weights=None):
    """

    :param sigmoided_logits:
    :param logits:
    :param ground_truth:
    :param n_iterations:
    :param iteration_weighing:
    :param use_vgg_loss:
    :param vgg_fmaps:
    :param vgg_weights:
    :return:
    """
    if iteration_weighing == 'increasing':
        z = 0.5 * (n_iterations * (n_iterations + 1))  # weighing normalization constant
    elif iteration_weighing == 'equal':
        z = n_iterations
    else:
        raise ValueError('iteration_weighing was {} and can be either equal or increasing'.format(iteration_weighing))

    assert (sigmoided_logits is not None and logits is not None)

    if use_vgg_loss:
        vgg_enc = VggEncoder(name='VGG19_Encoder', cutoff='vgg_19/conv4/conv4_1')
        if not (len(vgg_fmaps) == len(vgg_weights)) and vgg_weights is not None:
            raise ValueError('argument mismatch: '
                             'given {} vgg_fmaps and {} vgg_weights'.format(len(vgg_fmaps), len(vgg_weights)))
        annots_fmaps = vgg_enc(ground_truth,
                               purpose='loss',
                               requested_vgg_feature_maps=vgg_fmaps,
                               verbose=False)

    loss = tf.constant(0, dtype=tf.float32)
    loss_summaries = []

    for i in range(n_iterations):
        loss_i = tf.constant(0, dtype=tf.float32)
        # topo loss (optional)
        if use_vgg_loss:
            pred_fmaps = vgg_enc(sigmoided_logits[i],
                                 purpose='loss',
                                 requested_vgg_feature_maps=vgg_fmaps,
                                 verbose=False)

            loss_topo_i = tf.constant(0, dtype=tf.float32)
            for fm_pred, fm_gt, w in zip(pred_fmaps, annots_fmaps, vgg_weights):
                print('perceptual loss fmaps: {}--{}'.format(fm_pred, fm_gt))
                loss_topo_i = loss_topo_i + w * tf.reduce_mean((fm_pred - fm_gt) ** 2)

            topo_raw_i = tf.summary.scalar('topo-' + str(i + 1), loss_topo_i)
            loss_summaries.append(topo_raw_i)
            loss_i += loss_topo_i

        # bce loss
        loss_bce_i = sigmoid_cross_entropy_balanced(logits[i], ground_truth,
                                                    name='bce-' + str(i))

        bce_raw_i = tf.summary.scalar('bce' + str(i + 1), loss_bce_i)
        loss_summaries.append(bce_raw_i)
        loss_i += loss_bce_i

        if iteration_weighing == 'increasing':
            loss_i_weighted = ((i + 1) / z) * loss_i
        elif iteration_weighing == 'equal':
            loss_i_weighted = loss_i
        else:
            raise ValueError('iteration_weighing was {} and can be either equal or increasing'.format(iteration_weighing))

        loss += loss_i_weighted
        loss_i_weighted_raw = tf.summary.scalar('loss-after-' + str(i + 1), loss_i_weighted)

        loss_summaries.append(loss_i_weighted_raw)

    return loss, loss_summaries
