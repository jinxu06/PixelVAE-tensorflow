"""
The core Pixel-CNN model
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
import pixelcnn.nn_for_cond as nn

@add_arg_scope
def cond_pixel_cnn(x, gh=None, sh=None, nonlinearity=tf.nn.elu, dropout_p=0.5, nr_resnet=5, nr_filters=160, nr_logistic_mix=10, counters={}):
    name = nn.get_name("conv_pixel_cnn", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([nn.conv2d, nn.conv2d_1x1, nn.deconv2d, nn.gated_resnet, nn.dense], counters=counters, dropout_p=dropout_p):
            with arg_scope([nn.gated_resnet], nonlinearity=nonlinearity, gh=gh, sh=sh):

                xs = nn.int_shape(x)
                x_pad = tf.concat([x,tf.ones(xs[:-1]+[1])],3) # add channel of ones to distinguish image from padding later on

                u_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 3]))] # stream for pixels above
                ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1,3])) + \
                           nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2,1]))] # stream for up and to the left
                for rep in range(nr_resnet):
                    u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
                    ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))

                x_out = nn.nin(tf.nn.elu(ul_list[-1]), 10*nr_logistic_mix)
                return x_out
