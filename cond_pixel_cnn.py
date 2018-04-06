"""
The core Pixel-CNN model
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
import pixelcnn.nn_for_cond as nn

@add_arg_scope
def cond_pixel_cnn(x, gh=None, sh=None, nonlinearity=tf.nn.elu, dropout_p=0.0, nr_resnet=5, nr_filters=160, nr_logistic_mix=10, counters={}):
    name = nn.get_name("conv_pixel_cnn", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([nn.conv2d, nn.deconv2d, nn.gated_resnet, nn.dense], counters=counters, dropout_p=dropout_p):
            with arg_scope([nn.gated_resnet], nonlinearity=nonlinearity, gh=gh, sh=sh):
                xs = nn.int_shape(x)
                x_pad = tf.concat([x,tf.ones(xs[:-1]+[1])],3) # add channel of ones to distinguish image from padding later on

                u_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 3]))] # stream for pixels above
                ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1,3])) + \
                           nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2,1]))] # stream for up and to the left
                receptive_field = (2, 3)
                for rep in range(nr_resnet):
                    u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
                    ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))
                    receptive_field = (receptive_field[0]+1, receptive_field[1]+2)
                x_out = nn.nin(tf.nn.elu(ul_list[-1]), 10*nr_logistic_mix)
                print("    * receptive_field", receptive_field)
                return x_out

@add_arg_scope
def mix_logistic_sampler(params, nr_logistic_mix=10, sample_range=3., counters={}):
    name = nn.get_name("logistic_mix_sampler", counters)
    print("construct", name, "...")
    epsilon = 1. / ( tf.exp(sample_range)+1. )
    x = nn.sample_from_discretized_mix_logistic(params, nr_logistic_mix, epsilon)
    return x

@add_arg_scope
def mix_logistic_loss(x, params, masks=None):
    l = nn.discretized_mix_logistic_loss(x, params, sum_all=True, masks=masks)
    return l
