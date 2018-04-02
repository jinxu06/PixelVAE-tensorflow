"""
The core Pixel-CNN model
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
import pixel_cnn_pp.nn as nn

def deep_pixel_cnn(x, gh=None, sh=None, init=False, ema=None, dropout_p=0.5, nr_resnet=5, nr_filters=160, nr_logistic_mix=10):
    print("Use Deep PixelCNN")
    with tf.variable_scope('deep_pixel_cnn'):
        counters = {}
        with arg_scope([nn.conv2d, nn.conv2d_1x1, nn.deconv2d, nn.gated_resnet, nn.dense], counters=counters, init=init, ema=ema, dropout_p=dropout_p):

            resnet_nonlinearity = tf.nn.elu
            # sh = nn.latent_deconv_net(sh, scale_factor=1)

            with arg_scope([nn.conv2d], nonlinearity=resnet_nonlinearity):
                # sh = nn.conv2d(sh, 2*nr_filters, filter_size=[3,3], stride=[1,1], pad='SAME')
                # sh = nn.conv2d(sh, 2*nr_filters, filter_size=[3,3], stride=[1,1], pad='SAME')
                sh_2 = nn.conv2d(sh, nn.int_shape(sh)[-1], filter_size=[3,3], stride=[2,2], pad='SAME')
                sh_4 = nn.conv2d(sh_2, nn.int_shape(sh)[-1], filter_size=[3,3], stride=[2,2], pad='SAME')

            with arg_scope([nn.gated_resnet], nonlinearity=resnet_nonlinearity, gh=gh, sh=sh):

                # ////////// up pass through pixelCNN ////////
                xs = nn.int_shape(x)
                x_pad = tf.concat([x,tf.ones(xs[:-1]+[1])],3) # add channel of ones to distinguish image from padding later on
                u_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2, 3]))] # stream for pixels above
                ul_list = [nn.down_shift(nn.down_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[1,3])) + \
                           nn.right_shift(nn.down_right_shifted_conv2d(x_pad, num_filters=nr_filters, filter_size=[2,1]))] # stream for up and to the left
                for rep in range(nr_resnet):
                    u_list.append(nn.gated_resnet(u_list[-1], conv=nn.down_shifted_conv2d))
                    ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], conv=nn.down_right_shifted_conv2d))

                u_list.append(nn.down_shifted_conv2d(u_list[-1], num_filters=nr_filters, stride=[2, 2]))
                ul_list.append(nn.down_right_shifted_conv2d(ul_list[-1], num_filters=nr_filters, stride=[2, 2]))
                for rep in range(nr_resnet):
                    u_list.append(nn.gated_resnet(u_list[-1], sh=sh_2, conv=nn.down_shifted_conv2d))
                    ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], sh=sh_2, conv=nn.down_right_shifted_conv2d))

                u_list.append(nn.down_shifted_conv2d(u_list[-1], num_filters=nr_filters, stride=[2, 2]))
                ul_list.append(nn.down_right_shifted_conv2d(ul_list[-1], num_filters=nr_filters, stride=[2, 2]))
                for rep in range(nr_resnet):
                    u_list.append(nn.gated_resnet(u_list[-1], sh=sh_4, conv=nn.down_shifted_conv2d))
                    ul_list.append(nn.gated_resnet(ul_list[-1], u_list[-1], sh=sh_4, conv=nn.down_right_shifted_conv2d))

                # remember nodes
                for t in u_list+ul_list:
                    tf.add_to_collection('checkpoints', t)

                # /////// down pass ////////
                u = u_list.pop()
                ul = ul_list.pop()
                for rep in range(nr_resnet):
                    u = nn.gated_resnet(u, u_list.pop(), sh=sh_4, conv=nn.down_shifted_conv2d)
                    ul = nn.gated_resnet(ul, tf.concat([u, ul_list.pop()],3), sh=sh_4, conv=nn.down_right_shifted_conv2d)
                    tf.add_to_collection('checkpoints', u)
                    tf.add_to_collection('checkpoints', ul)

                u = nn.down_shifted_deconv2d(u, num_filters=nr_filters, stride=[2, 2])
                ul = nn.down_right_shifted_deconv2d(ul, num_filters=nr_filters, stride=[2, 2])
                for rep in range(nr_resnet+1):
                    u = nn.gated_resnet(u, u_list.pop(), sh=sh_2, conv=nn.down_shifted_conv2d)
                    ul = nn.gated_resnet(ul, tf.concat([u, ul_list.pop()],3), sh=sh_2, conv=nn.down_right_shifted_conv2d)
                    tf.add_to_collection('checkpoints', u)
                    tf.add_to_collection('checkpoints', ul)

                u = nn.down_shifted_deconv2d(u, num_filters=nr_filters, stride=[2, 2])
                ul = nn.down_right_shifted_deconv2d(ul, num_filters=nr_filters, stride=[2, 2])
                for rep in range(nr_resnet+1):
                    u = nn.gated_resnet(u, u_list.pop(), conv=nn.down_shifted_conv2d)
                    ul = nn.gated_resnet(ul, tf.concat([u, ul_list.pop()],3), conv=nn.down_right_shifted_conv2d)
                    tf.add_to_collection('checkpoints', u)
                    tf.add_to_collection('checkpoints', ul)

                x_out = nn.nin(tf.nn.elu(ul),10*nr_logistic_mix)

                assert len(u_list) == 0
                assert len(ul_list) == 0

                return x_out
