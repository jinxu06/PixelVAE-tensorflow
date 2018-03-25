"""
The core Pixel-CNN model
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
import pixel_cnn_pp.nn as nn

def model_spec(x, gh=None, sh=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=160, nr_logistic_mix=10, resnet_nonlinearity='concat_elu', energy_distance=False, global_conditional=False, spatial_conditional=False):
    """
    We receive a Tensor x of shape (N,H,W,D1) (e.g. (12,32,32,3)) and produce
    a Tensor x_out of shape (N,H,W,D2) (e.g. (12,32,32,100)), where each fiber
    of the x_out tensor describes the predictive distribution for the RGB at
    that position.
    'h' is an optional N x K matrix of values to condition our generative model on
    """

    counters = {}
    with arg_scope([nn.conv2d, nn.conv2d_1x1, nn.deconv2d, nn.gated_resnet, nn.dense], counters=counters, init=init, ema=ema, dropout_p=dropout_p):

        # parse resnet nonlinearity argument
        if resnet_nonlinearity == 'concat_elu':
            resnet_nonlinearity = nn.concat_elu
        elif resnet_nonlinearity == 'elu':
            resnet_nonlinearity = tf.nn.elu
        elif resnet_nonlinearity == 'relu':
            resnet_nonlinearity = tf.nn.relu
        else:
            raise('resnet nonlinearity ' + resnet_nonlinearity + ' is not supported')

        if spatial_conditional:
            if type(sh)==list:
                sh, sh_2, sh_4 = sh
            else:
                sh = nn.latent_deconv_net(sh, scale_factor=1)
                sh_2 = nn.conv2d(sh, nn.int_shape(sh)[-1], filter_size=[3,3], stride=[2,2], pad='SAME')
                sh_4 = nn.conv2d(sh_2, nn.int_shape(sh)[-1], filter_size=[3,3], stride=[2,2], pad='SAME')
        else:
            sh_2, sh_4 = None, None

        with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, gh=gh, sh=sh):

            ## Mini Conv

            # ////////// up pass through pixelCNN ////////
            xs = nn.int_shape(x)
            x_pad = tf.concat([x,tf.ones(xs[:-1]+[1])],3) # add channel of ones to distinguish image from padding later on


            u_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 3]))] # stream for pixels above
            ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1,3])) + \
                       nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2,1]))] # stream for up and to the left
            for rep in range(nr_resnet):
                u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
                ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))

            for t in u_list+ul_list:
                tf.add_to_collection('checkpoints', t)

            x_out = nn.nin(tf.nn.elu(ul_list[-1]),10*nr_logistic_mix)
            return x_out
