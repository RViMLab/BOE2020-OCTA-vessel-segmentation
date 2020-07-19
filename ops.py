from __future__ import division
import tensorflow as tf
import tensorflow.contrib as tf_contrib


def conv2d(x, w, padding='SAME', strides=[1, 1, 1, 1], keep_prob=1.0, verbose=True):
    """
    :param x: input tensor of shape (B,H,W,C)
    :param w: kernel weights of shape [kernel size, kernel size, incoming_feature_maps, filters]
    :param padding: 'SAME' to preserve feature maps dimension, look tf docs for other options
    :param strides: strides along every dimension of the input tensor. If all 1 then no downsampling due to striding
    :param keep_prob: must equal 1-p where p is the dropout probability
    :param verbose: True to print conv layer settings False to supress
    :return: out: output tensor passed through dropout
    """
    x_shape = x.get_shape()
    w_shape = w.get_shape().as_list()
    out = tf.nn.conv2d(x, w, strides=strides, padding=padding)
    out_shape = out.get_shape()

    if verbose:
        print(' Normal_2D_Convolution with s =', 1, ' k =', w_shape[0], ' p = SAME')
        print('    W:', w_shape)
        print('    in:', x_shape)
        print('    out:', out_shape)

    return tf.nn.dropout(out, keep_prob=keep_prob)


def dilated_conv2d(x, w, dilation_rate, keep_prob=1.0, verbose=True):
    """G
    This op assumes static shapes H,W known in advance

    :param x: input tensor of shape (B,H,W,C)
    :param w: kernel weights of shape [kernel size, kernel size, channels, features]
    :param dilation_rate: dilation rate , if r = 1 the convolution acts as a classic convolution
    :param keep_prob: must equal 1-p where p is the dropout probability
    :param verbose: True to print conv layer settings False to supress
    :return: out: output tensor passed through dropout
    """
    x_shape = x.get_shape()
    w_shape = w.get_shape()

    out = tf.nn.convolution(x, w, dilation_rate=[dilation_rate, dilation_rate], padding='SAME')
    out_shape = out.get_shape()
    if verbose:
        print('Dilated_2D_Convolution with s =', 1, 'r =', dilation_rate, ' k =', w_shape[0], ' p = SAME',
              ' dropout rate = ', 1 - keep_prob)
        print('    W:', w_shape)
        print('    in:', x_shape)
        print('    out:', out_shape)
    return tf.nn.dropout(out, keep_prob=1.0)


def transposed_conv2d(x, w, stride, keep_prob=1.0, verbose=True):
    """
    This op  does not assume static shapes H,W known in advance

    :param x: input tensor of shape (B,H,W,C)
    :param w: kernel weights of shape [height, width, output_channels, in_channels] !SOS at the order of shape elements!
    :param stride: stride is the factor by which the output resolution will be increased.
                   typically stride=2 will lead to 2x upsamples
    :param keep_prob: must equal 1-p where p is the dropout probability
    :param verbose: True to print conv layer settings False to supress
    :return: out: output tensor passed through dropout
    """
    x_shape = tf.shape(x)
    w_shape = tf.shape(w)

    output_shape = tf.stack([x_shape[0], x_shape[1] * stride, x_shape[2] * stride, x_shape[3] // 2])
    out = tf.nn.conv2d_transpose(x, w, output_shape, strides=[1, stride, stride, 1], padding='SAME')
    out_shape = tf.shape(out)

    if verbose:
        print(' Transposed_Convolution with s =', stride, ' k =', w_shape[0], ' p = SAME')
        print('    W:', w_shape[0])
        print('    in:', x_shape)
        print('    out:', out_shape)

    return tf.nn.dropout(out, keep_prob=keep_prob)


def max_pool(x, n, verbose=False):
    """
    :param x: input tensor of shape (B,H,W,C)
    :param n: the pooling kernel size
    :param verbose: True to print conv layer settings False to supress
    :return: out: out_H = ceil[ (H-n) / s ] + 1, out_W = ceil[ (W-n) / s ]
    """
    x_shape = x.get_shape()

    out = tf.nn.max_pool(x, ksize=[1, n, n, 1], strides=[1, n, n, 1], padding='SAME')
    out_shape = out.get_shape()
    if verbose:
        print('    in:', x_shape)
        print(' Max Pooling with s =', n, ' k =', n, ' p = SAME')
        print('    out:', out_shape)
    return out


def crop_and_concat(x1, x2):
    """
    crops tensor x1 to be of equal spatial dimensions as x2 and concats them along the featurmap axis
    :param x1: tensor (B,H1,W1,C1)
    :param x2: tensor (B,H2,W2,C2)
    :return: concatenated tensor of shape (B,H2,W2,C1+C2)
    """
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # print('in crop concat')
    # print(x1.shape)
    # print(x2.shape)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
    size = [-1, x2_shape[1], x2_shape[2], -1]
    x1_crop = tf.slice(x1, offsets, size)
    out = tf.concat([x1_crop, x2], 3)
    # print(out.shape)
    return out


def resolve_shape(tensor, rank=None, scope=None):
    """Fully resolves the shape of a Tensor.
      Use as much as possible the shape components already known during graph
      creation and resolve the remaining ones during runtime.
      Args:
        tensor: Input tensor whose shape we query.
        rank: The rank of the tensor, provided that we know it.
        scope: Optional name scope.
      Returns:
        shape: The full shape of the tensor.
    """
    with tf.name_scope(scope, 'resolve_shape', [tensor]):
        if rank is not None:
            shape = tensor.get_shape().with_rank(rank).as_list()
        else:
            shape = tensor.get_shape().as_list()
        if None in shape:
            shape_dynamic = tf.shape(tensor)
            for i in range(len(shape)):
                if shape[i] is None:
                    shape[i] = shape_dynamic[i]
        return shape


def bias_variable(name, shape, seed=1):
    var = tf.get_variable(name, shape,
                          initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed), dtype=tf.float32)
    return var


def weight_variable(name, shape, seed=1):
    var = tf.get_variable(name, shape=shape,
                          initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=seed), dtype=tf.float32)
    return var


def res_block(x, num_feature_maps, prev_num_channels=None, filter_size=3, dilation_rate=2,
              bn_train=True, norm_type='bn', keep_prob=1.0,
              do_dropout_in_last_encoder_layer=False, verbose=False):

    """
    adds a residual block of architecture: x ------> conv-bn-relu --> Conv-bn -------->(+) --> relu
                                               |                                        |
                                                ---------------conv-bn-relu-------------
    as described by the original Resnet paper. Note: there are alternative designs not used here
    :param x: input tensor of shape (B,H,W,C)
    :param num_feature_maps: the number of feature maps in each conv layer of the residual block
    :param prev_num_channels: the number of feature maps in the input tensor
    :param filter_size: the filter size of each conv filter
    :param dilation_rate: dilation rate of each convolution layer if using dilated convolution
    :param bn_train: bn switch, if True then updates moving mean/var and uses current batch statistics for normalization
                              if False then does not update moving mean/var and uses them for normalization
    :param norm_type: 'bn', 'in', 'nn', 'gn'
    :param keep_prob: 1-p where p is the dropout propability for using dropout after the convolution
    :param do_dropout_in_last_encoder_layer: if True adds dropout with keep_prob only at after the last conv layer
    :param verbose: if True prints information on the conv layers input, output, kernel, dilation etc...
    :return: output: tensor output of residual block
    """
    if not (norm_type == 'bn' or norm_type == 'gn'):
        raise ValueError('norm_type [{}] is invalide use bn or gn'.format(norm_type))

    if prev_num_channels is None:
        prev_num_channels = x.shape[3]

    if verbose:
        print('In Res-block')
        print('prev_num_channels', prev_num_channels)

    if do_dropout_in_last_encoder_layer and verbose:
        print('Doing Dropout here with keep_prob', keep_prob)
    else:
        keep_prob = 1.0

    # conv1
    b_1 = bias_variable("b1", [num_feature_maps])
    w_1 = weight_variable("w1", [filter_size, filter_size, prev_num_channels, num_feature_maps])
    conv_1 = conv2d(x, w_1, verbose=verbose) + b_1

    if norm_type == 'bn':
        # bn - relu
        bn_1 = tf.contrib.layers.batch_norm(inputs=conv_1, decay=0.9, is_training=bn_train, center=True, scale=True,
                                            activation_fn=tf.nn.relu, updates_collections=None, fused=True)

    elif norm_type == 'gn':
        # gn -relu
        bn_1 = group_norm(conv_1, g=32, scope='group-norm-1')
        bn_1 = tf.nn.relu(bn_1)

    # conv2
    b_2 = bias_variable("b2", [num_feature_maps])
    w_2 = weight_variable("w2", [filter_size, filter_size, num_feature_maps, num_feature_maps])
    conv_2 = conv2d(bn_1, w_2, verbose=verbose) + b_2

    if norm_type == 'bn':
        # bn
        bn_2 = tf.contrib.layers.batch_norm(inputs=conv_2, decay=0.9, is_training=bn_train, center=True, scale=True,
                                            activation_fn=None, updates_collections=None, fused=True)

    elif norm_type == 'gn':
        # gn
        bn_2 = group_norm(conv_2, g=32, scope='group-norm-2')

    # conv3
    # this conv layer makes the feature maps of x(input) equal to the number
    # of feature maps of bn_2(i.e the output of the second conv of the res block
    # if the x has the same number of feature maps as bn_2 then this conv layer is
    # skipped

    if prev_num_channels != num_feature_maps:

        b_s = bias_variable("bs", [num_feature_maps])
        w_s = weight_variable("ws", [filter_size, filter_size, prev_num_channels, num_feature_maps])
        shortcut = conv2d(x, w_s, verbose=verbose) + b_s

        if norm_type == 'bn':
            shortcut = tf.contrib.layers.batch_norm(inputs=shortcut, decay=0.9, is_training=bn_train, center=True,
                                                    scale=True, activation_fn=None, updates_collections=None,
                                                    fused=True)

        elif norm_type =='gn':

            shortcut = group_norm(shortcut, g=32, scope='group-norm-s')

    else:
        shortcut = x
    output = tf.nn.relu(shortcut + bn_2)
    return output


# Normalization layers
def batch_norm(x, is_training=False, scope='batch_norm'):
    return tf_contrib.layers.batch_norm(x,
                                        decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, renorm=True, updates_collections=None,
                                        is_training=is_training, scope=scope)


def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)


def layer_norm(x, scope='layer_norm'):
    return tf_contrib.layers.layer_norm(x,
                                        center=True, scale=True,
                                        scope=scope)


def group_norm(x, g=32, eps=1e-5, scope='group_norm'):
    with tf.variable_scope(scope):
        x_shape = tf.shape(x)
        n = x_shape[0]
        h = x_shape[1]
        w = x_shape[2]
        c = x.get_shape().as_list()[3]
        g = tf.minimum(g, c)

        x = tf.reshape(x, [n, h, w, g, c // g])
        mean, var = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
        x = (x - mean) / tf.sqrt(var + eps)

        gamma = tf.get_variable('gamma', [1, 1, 1, c], initializer=tf.constant_initializer(1.0), trainable=True)
        beta = tf.get_variable('beta', [1, 1, 1, c], initializer=tf.constant_initializer(0.0), trainable=True)

        x = tf.reshape(x, [n, h, w, c]) * gamma + beta

    return x



