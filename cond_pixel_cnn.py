"""
The core Pixel-CNN model
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
import pixelcnn.nn_for_cond as nn
from layers import down_shifted_conv2d, down_right_shifted_conv2d, down_shifted_deconv2d, down_right_shifted_deconv2d, down_shift, right_shift, gated_resnet, nin

@add_arg_scope
def cond_pixel_cnn(x, gh=None, sh=None, nonlinearity=tf.nn.elu, nr_resnet=5, nr_filters=160, nr_logistic_mix=10, bn=True, dropout_p=0.0, kernel_initializer=None, kernel_regularizer=None, is_training=False, counters={}):
    name = nn.get_name("conv_pixel_cnn", counters)
    print("construct", name, "...")
    print("    * nr_resnet: ", nr_resnet)
    print("    * nr_filters: ", nr_filters)
    print("    * nr_logistic_mix: ", nr_logistic_mix)
    with tf.variable_scope(name):
        # do not use batch normalization for auto-reggressive model, force bn to be False
        with arg_scope([gated_resnet], gh=gh, sh=sh, nonlinearity=nonlinearity, dropout_p=dropout_p, counters=counters):
            with arg_scope([gated_resnet, down_shifted_conv2d, down_right_shifted_conv2d, down_shifted_deconv2d, down_right_shifted_deconv2d], bn=False, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
                xs = nn.int_shape(x)
                x_pad = tf.concat([x,tf.ones(xs[:-1]+[1])],3) # add channel of ones to distinguish image from padding later on

                u_list = [down_shift(down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 3]))] # stream for pixels above
                ul_list = [down_shift(down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1,3])) + \
                        right_shift(down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2,1]))] # stream for up and to the left
                receptive_field = (2, 3)
                for rep in range(nr_resnet):
                    u_list.append(gated_resnet(u_list[-1], conv=down_shifted_conv2d))
                    ul_list.append(gated_resnet(ul_list[-1], u_list[-1], conv=down_right_shifted_conv2d))
                    receptive_field = (receptive_field[0]+1, receptive_field[1]+2)
                x_out = nin(tf.nn.elu(ul_list[-1]), 10*nr_logistic_mix)
                print("    * receptive_field", receptive_field)
                return x_out

@add_arg_scope
def mix_logistic_sampler(params, nr_logistic_mix=10, sample_range=3., counters={}):
    name = nn.get_name("logistic_mix_sampler", counters)
    print("construct", name, "...")
    epsilon = 1. / ( tf.exp(float(sample_range))+1. )
    x = nn.sample_from_discretized_mix_logistic(params, nr_logistic_mix, epsilon)
    return x

@add_arg_scope
def mix_logistic_loss(x, params, masks=None):
    l = nn.discretized_mix_logistic_loss(x, params, sum_all=False, masks=masks)
    return tf.reduce_mean(l)
