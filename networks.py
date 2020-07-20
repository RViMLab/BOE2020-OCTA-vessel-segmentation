import tensorflow as tf
from collections import OrderedDict
from ops import *


def enc_dec_network_def(x, n_heatmaps=1, norm_layer_type='bn', net_id='',
                        feature_maps_root=64,
                        layers=3,
                        conv_kernel_size=3,
                        pool_kernel_size=2,
                        bn_in_train_mode=True,
                        seed=1,
                        do_dropout_in_encoder=False,
                        keep_prob=1.0,
                        verbose=False):
    """
    defines the graph for a encoder-decoder network. Both the encoder and decoder use residual blocks instead
    of simple conv layers.
    -It supports adding multiple distinct encoder-decoder nets using the net_id string argument which is used
     to distinguish between variable scopes of different modules.
    -In iUNET the net_id is always kept as an empty string to add a module to the graph with shared weights

    :param x: input tensor of shape (B,H,W,C)
    :param n_heatmaps: the channel dimension of the output tensor (for binary with sigmoid activation it is 1)
    :param norm_layer_type: normalization layer type
    :param net_id: used to control variable scope of identical networks with distinct weights (as in stacked networks)
    :param feature_maps_root: number of filters in the 1st conv layer of the unet
    :param layers: number of residual blocks in the encoder (equal number of blocks in the decoder)
                   Pooling is performed after each residual block.
    :param conv_kernel_size: convolutional kernel size
    :param pool_kernel_size: pooling kernel size
    :param bn_in_train_mode: if True use current batch statistics (as in training) else use saved dataset statistics
    :param seed: random seed for reproducible initialization
    :param do_dropout_in_encoder: if True does dropout in the inner most encoder
    :param keep_prob:
    :param verbose: if True prints staff while defining the network
    :return:
    """

    in_node = x   # in_node is always the input to the next layer
    pools = OrderedDict()  # stores all pooling layer outputs
    deconv = OrderedDict()  # stores all transposed conv layer outputs
    enc_convs = OrderedDict()  # stores all conv layer outputs of the ENCODER
    dec_convs = OrderedDict()  # stores all transposed conv layer outputs of the DECODER
    do_dropout_in_last_encoder_layer = False  # this will only change if do_dropout_in_encoder=True:

    # ENCODER layers
    for layer in range(0, layers):
        with tf.variable_scope('down' + str(layer) + net_id):
            if verbose:
                print('down' + str(layer) + net_id)
            feature_maps = 2 ** layer * feature_maps_root

            if layer == layers - 1 and do_dropout_in_encoder:
                do_dropout_in_last_encoder_layer = True
                if verbose:
                    print('Doing dropout after last encoder conv layer')

            if layer == 0:
                # print(layer)
                enc_convs[layer] = res_block(in_node,
                                             num_feature_maps=feature_maps,
                                             filter_size=conv_kernel_size,
                                             bn_train=bn_in_train_mode, norm_type=norm_layer_type,
                                             verbose=verbose)

            else:
                enc_convs[layer] = res_block(in_node,
                                             num_feature_maps=feature_maps,
                                             filter_size=conv_kernel_size,
                                             bn_train=bn_in_train_mode, norm_type=norm_layer_type,
                                             keep_prob=keep_prob,
                                             do_dropout_in_last_encoder_layer=do_dropout_in_last_encoder_layer,
                                             verbose=verbose)

            if layer < layers - 1:
                pools[layer] = max_pool(enc_convs[layer], pool_kernel_size)
                in_node = pools[layer]

    # this is the final output of the decoder that is fed towards the DECODER
    in_node = enc_convs[layers - 1]

    # DECODER layers - with skip connections
    for layer in range(layers - 2, -1, -1):
        with tf.variable_scope('up' + str(layer) + net_id):
            if verbose:
                print('up' + str(layer) + net_id)

            feature_maps = 2 ** (layer + 1) * feature_maps_root

            wd = weight_variable("wd"+net_id, [pool_kernel_size, pool_kernel_size, feature_maps // 2, feature_maps],
                                 seed=seed)

            bd = bias_variable("bd" + net_id, [feature_maps // 2], seed=seed)

            # transposed conv
            h_deconv = transposed_conv2d(in_node, wd, pool_kernel_size, verbose=verbose) + bd

            # crop and concat with feature maps from encoder
            # layer with the same (or closest to) resolution
            # as that of the transposed convolution layer
            h_deconv_concat = crop_and_concat(enc_convs[layer], h_deconv)

            deconv[layer] = h_deconv_concat

            in_node = res_block(h_deconv_concat,
                                num_feature_maps=feature_maps//2,
                                prev_num_channels=feature_maps,
                                filter_size=conv_kernel_size,
                                bn_train=bn_in_train_mode, norm_type=norm_layer_type,
                                verbose=verbose)

            dec_convs[layer] = in_node

    # final conv layer
    with tf.variable_scope('output' + net_id):
        weight = weight_variable("w" + net_id, [1, 1, feature_maps_root, n_heatmaps], seed=seed)
        bias = bias_variable("b" + net_id, [n_heatmaps], seed=seed)
        conv = conv2d(in_node, weight, verbose=verbose)  # keep_prob=keep_prob)
        logits = conv + bias
        dec_convs["out"] = logits
    return logits, enc_convs, dec_convs  # logits is the linear output from the final conv layer of the network


def unet(x, n_heatmaps=1, num_layers=3, feature_maps_root=64, norm_type='bn',
         batch_norm_in_train_mode=True,
         name='UNET', verbose=False):
    """
    defines a unet (based on enc_dec_network_def to do it)
    see enc_dec_network_def for an explanation of the arguments
    """
    with tf.variable_scope(name):
        logits, encoder, decoder = enc_dec_network_def(x=x,
                                                       n_heatmaps=n_heatmaps,
                                                       bn_in_train_mode=batch_norm_in_train_mode,
                                                       layers=num_layers,
                                                       norm_layer_type=norm_type,
                                                       feature_maps_root=feature_maps_root,
                                                       verbose=verbose)
        # out_activation == 'sigmoid':
        sigmoided_logits = tf.nn.sigmoid(logits)

    return sigmoided_logits, logits, encoder, decoder


def iunet(x, n_heatmaps=1, num_layers=3, feature_maps_root=64, norm_type='bn',
          batch_norm_in_train_mode=True,
          n_iterations=4,
          name='iUNET', verbose=False):
    """
    defines a iUNET (based on enc_dec_network_def to do it)
    see enc_dec_network_def for an explanation of most arguments
    n_iterations : refers to the number of refinement iterations the model performs
    """

    # lists of logits and sigmoid-ed logits, one for each iteration
    logits = []
    sigmoided_logits = []
    with tf.variable_scope(name) as scope:
        for it in range(n_iterations):
            with tf.name_scope('input_{}'.format(it)):
                if it == 0:
                    shape = tf.shape(x)  # (None,h,w,c)
                    heatmap_0 = tf.zeros(shape=shape, dtype=tf.float32)
                    x_i = tf.concat([x, heatmap_0], axis=3)
                    if verbose:
                        print('inputs_0:', x_i.shape)
                else:
                    heatmap_prev = sigmoided_logits[it - 1]
                    x_i = tf.concat([x, heatmap_prev], axis=3)

            logits_i, encoder, decoder = enc_dec_network_def(x=x_i,
                                                             n_heatmaps=n_heatmaps,
                                                             bn_in_train_mode=batch_norm_in_train_mode,
                                                             layers=num_layers,
                                                             norm_layer_type=norm_type,
                                                             feature_maps_root=feature_maps_root,
                                                             verbose=verbose)

            scope.reuse_variables()  # sharing network across iterations
            logits.append(logits_i)
            sigmoided_logits.append(tf.nn.sigmoid(logits_i))

    return sigmoided_logits, logits, encoder, decoder


def shn(x, n_heatmaps=1, num_layers=3, feature_maps_root=64, norm_type='bn',
        batch_norm_in_train_mode=True,
        n_modules=4,
        name='SHN', verbose=False):
    """
    defines a SHN (based on enc_dec_network_def to do it)
    see enc_dec_network_def for an explanation of most arguments
    n_modules : refers to the number of stacked modules the model consists of
    """

    # lists of logits and sigmoid-ed logits, one for each module
    logits = []
    sigmoided_logits = []
    with tf.variable_scope(name):
        for module in range(n_modules):
            with tf.name_scope('input_{}'.format(module)):
                if module == 0:
                    shape = tf.shape(x)
                    heatmap_0 = tf.zeros(shape=shape, dtype=tf.float32)
                    x_i = tf.concat([x, heatmap_0], axis=3)
                    if verbose:
                        print('inputs_0:', x_i.shape)
                else:
                    heatmap_prev = sigmoided_logits[module - 1]
                    x_i = tf.concat([x, heatmap_prev], axis=3)

            logits_i, encoder, decoder = enc_dec_network_def(x=x_i,
                                                             n_heatmaps=n_heatmaps,
                                                             bn_in_train_mode=batch_norm_in_train_mode,
                                                             layers=num_layers,
                                                             norm_layer_type=norm_type,
                                                             feature_maps_root=feature_maps_root,
                                                             verbose=verbose,
                                                             net_id=str(module))
            logits.append(logits_i)
            sigmoided_logits.append(tf.nn.sigmoid(logits_i))

    return sigmoided_logits, logits, encoder, decoder
