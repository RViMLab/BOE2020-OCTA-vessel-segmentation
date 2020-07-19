import tensorflow as tf
import os
import cv2
import numpy as np
from skimage.morphology import skeletonize


def save_2d_segmentation(prediction, filename, save_dir_path):
    """
    :param prediction: sigmoided prediction of shape (1,H,W,1)
    :param filename: name of 2d file
    :param save_dir_path: path to dir where the prediction file will be saved
    :return: flag (the return value if imwrite)
    """
    # some checks
    if not len(prediction.shape) == 4:
        raise ValueError('expects prediction to be of shape (1,H,W,1) '
                         'and instead got shape{}'.format(prediction.shape))
    if prediction.shape[0] > 1:
        raise ('expects prediction to be of shape (1,H,W,1) '
               'and instead got shape {}'.format(prediction.shape))

    flag = cv2.imwrite(os.path.join(save_dir_path, filename), img=prediction[0, :, :, :] * 255)
    return flag


def select_ckpt_file(model_dir):

    """
    :param model_dir:
    return
    """
    files_in_model_dir = os.listdir(model_dir)
    files_in_model_dir = [x for x in files_in_model_dir if 'index' in x]
    ckpt_file_path = files_in_model_dir[0]
    ckpt_file_path = ckpt_file_path[:-6]
    return os.path.join(model_dir, ckpt_file_path)


# metrics
def project_2dpoint_on_segment(rst, cst, point):
    """
    finds minimum distance between point and set of 2d-points
    :param rst: rows of set of points
    :param cst: cols of set of points
    :param point: [row,col]
    :return:
    """
    #row = point[0]
    #col = point[1]
    point = np.expand_dims(point, axis=0)

    rst = np.expand_dims(rst, axis=1)

    cst = np.expand_dims(cst, axis=1)

    #print(point.shape)
    #print(rst.shape)
    #print(cst.shape)
    vs_rc = point-np.hstack((rst, cst))
    dists = np.sum(np.abs(vs_rc)**2, axis=-1)**(1./2)
    # print(np.sum(dists==0.0))

    # dists = dists[dists != 0] # commented that for debugging!

    # print(np.sum(dists==0.0))
    dists_min = dists.min()
    return dists_min


def correctness_completeness_quality(ground_truth_list, propability_maps_list, theta=2, threshold=0.5, verbose=False):
    """

    :param ground_truth_list: list of ground truth segmentations (do not have to be skeletonized)
                              each of shape (1,H,W,C) (for OCTA data C=1)
    :param propability_maps_list: list of predicted segmentation heatmaps (do not have to be skeletonized)
                              each of shape (1,H,W,C) (for OCTA data C=1)
    :param theta: the tolerance expressed in pixels for comparing skeletonized predictions.
    :param threshold: binarization threshold for making heatmaps binary before skeletonizing them
    :param verbose: if True prints messages for debugging.
    :return: corr, complet, quality metrics (mean across the elements of the ground_truth_list)
             also returns qaulities which correspond to the quality of each element of the ground_truth_list
    """
    num_images = len(propability_maps_list)
    assert(len(propability_maps_list) == len(ground_truth_list))

    completeness_sum = 0
    correctness_sum = 0
    quality_sum = 0

    qualities = []  # this contains the quality score of each prediction in the propability_maps_list
    correctnesses = []
    completenesses = []
    for propability_map, y_gt in zip(propability_maps_list, ground_truth_list):

        y_hat = np.copy(propability_map[0, :, :, 0])

        idx = y_hat > threshold
        idx_n = y_hat <= threshold
        y_hat[idx] = 1.000
        y_hat[idx_n] = 0.000

        y = np.copy(y_gt[0, :, :, 0])

        y = np.array(y, dtype=np.uint8)
        skeleton_gt = skeletonize(y > 0.0)
        skeleton_gt = np.array(skeleton_gt, dtype=np.uint8)

        y_hat = np.array(y_hat, dtype=np.uint8)
        skeleton_pred = skeletonize(y_hat > 0)
        skeleton_pred = np.array(skeleton_pred, dtype=np.uint8)

        X = np.copy(skeleton_gt)
        P = np.copy(skeleton_pred)

        # indices of positive points on ground_truth skeleton
        GT_inds_tuple = np.where(X > 0)
        r_gt = GT_inds_tuple[0]
        c_gt = GT_inds_tuple[1]
        cardinality_X = len(r_gt)

        # indices of positive points on prediction skeleton
        Pred_inds_tuple = np.where(P > 0)
        r_pred = Pred_inds_tuple[0]
        c_pred = Pred_inds_tuple[1]
        cardinality_P = len(r_pred)

        #print(cardinality_P, cardinality_X)

        completeness_nominator = 0
        for i in range(cardinality_X):

            point = np.array([r_gt[i], c_gt[i]])
            dists_min = project_2dpoint_on_segment(r_pred, c_pred, point)
            if dists_min < theta:  # changed to equal
                completeness_nominator += 1

        completeness = completeness_nominator / cardinality_X

        correctness_nominator = 0
        for i in range(cardinality_P):
            point = np.array([r_pred[i], c_pred[i]])
            dists_min = project_2dpoint_on_segment(r_gt, c_gt, point)
            if dists_min < theta:  # changed to equal
                correctness_nominator += 1

        correctness = correctness_nominator / cardinality_P

        quality = correctness_nominator / (cardinality_P - completeness_nominator + cardinality_X)

        completeness_sum += completeness
        correctness_sum += correctness
        quality_sum += quality
        qualities.append(quality)
        correctnesses.append(correctness)
        completenesses.append(completeness)
        # print(qualities)
    return correctness_sum / num_images, completeness_sum / num_images, quality_sum / num_images, qualities, correctnesses, completenesses


#  tensorflow based utilities  #
def identity_on_tensors(tensor1, tensor2):
    return tensor1, tensor2


def track_mean_value_per_epoch(i, loss_value, epoch_step, epoch, values_list, summary_writer, tag='loss-per-epoch'):
    """ utility function for tracking mean loss
    :param i: iteration
    :param loss_value: loss_value @ i
    :param epoch_step: num of iterations in an epoch
    :param epoch: epoch count
    :param values_list: list of loss_values for current epoch
    :param summary_writer: summary writer object
    :param tag: tag of summary value
    :return: updated epoch count, values_list, summary writer (with summary of mean loss added if epoch elapsed)
    """
    cost_mean = np.inf
    if not i == 0:
        if i % epoch_step == 0:
            epoch += 1
            cost_mean = np.mean(np.array(values_list))
            print('Epoch [{}]: mean loss_T : {}'.format(epoch, cost_mean))
            values_list = []
            summary = tf.Summary()
            summary.value.add(tag=tag, simple_value=cost_mean)
            summary_writer.add_summary(summary, i)
        else:
            values_list.append(loss_value)

    return epoch, values_list, cost_mean, summary_writer


def count_records_in_tfrecord(tfrecord_filename):
    cnt = 0
    for record in tf.python_io.tf_record_iterator(tfrecord_filename):
        cnt += 1
    assert(cnt > 0)
    return cnt


def make_learning_rate_sceduler(lr_schedule_type, lr_start, global_step, decay_steps=None, start_decay_step=None,
                                end_learning_rate=None, decay_rate=None):
    with tf.name_scope(lr_schedule_type + '_decay'):
        if lr_schedule_type == 'linear':
            learning_rate = (
                tf.where(tf.greater_equal(global_step, start_decay_step),
                         tf.train.polynomial_decay(lr_start, global_step - start_decay_step,
                                                   decay_steps, end_learning_rate, power=1.0), lr_start))

        elif lr_schedule_type == 'polynomial_order_2':
            learning_rate = (
                tf.where(tf.greater_equal(global_step, start_decay_step),
                         tf.train.polynomial_decay(lr_start, global_step - start_decay_step,
                                                   decay_steps,
                                                   end_learning_rate, power=2.0), lr_start))
        elif lr_schedule_type == 'inverse_time':
            learning_rate = tf.train.inverse_time_decay(learning_rate=lr_start,
                                                        global_step=global_step,
                                                        decay_steps=decay_steps,
                                                        decay_rate=decay_rate,
                                                        staircase=True)
        elif lr_schedule_type == 'exponential':
            raise NotImplementedError

        learning_rate_summary = tf.summary.scalar('learning_rate'.format(lr_schedule_type + '_decay'), learning_rate)
        return learning_rate, learning_rate_summary


def make_optimizer(loss, variables,
                   lr_scheduler_type,
                   lr_start,
                   decay_steps,
                   decay_rate=None,
                   start_decay_step=None,
                   end_learning_rate=None,
                   use_clip_gradients=False,
                   clip_value=100.0,
                   name='Adam'):
    with tf.name_scope(name):
        global_step = tf.Variable(0, trainable=False)
        learning_rate, learning_rate_summary = make_learning_rate_sceduler(lr_schedule_type=lr_scheduler_type,
                                                                           global_step=global_step,
                                                                           lr_start=lr_start,
                                                                           decay_steps=decay_steps,
                                                                           decay_rate=decay_rate,
                                                                           start_decay_step=start_decay_step,
                                                                           end_learning_rate=end_learning_rate)

        optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads_and_vars = optim.compute_gradients(loss, var_list=variables)  # list of tuples of (grad, var)
        if use_clip_gradients and clip_value is not None:
            capped_gvs = [(tf.clip_by_value(grad, -clip_value, clip_value), var) for grad, var in grads_and_vars]
            grads_and_vars = capped_gvs
        train_op = optim.apply_gradients(grads_and_vars, global_step=global_step)

        with tf.name_scope('grad_var_summaries'):
            grads_and_vars_summaries_list = []
            for grad, var in grads_and_vars:
                # this handles the possibility of using batch norm where the grad of moving mean/var variables is None
                # rather than a tensor
                if grad is not None:
                    print(var.name + '/gradient')
                    grads_and_vars_summaries_list.append(tf.summary.histogram(var.name + '/gradient', grad))
                    grads_and_vars_summaries_list.append(tf.summary.histogram(var.name, var))

            grads_and_vars_summary = tf.summary.merge(grads_and_vars_summaries_list)

    return train_op, learning_rate_summary, grads_and_vars_summary