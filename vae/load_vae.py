import numpy as np
import os
import tensorflow as tf
import argparse
import json
import time
import data.celeba_data as celeba_data
from tensorflow.contrib.framework.python.ops import arg_scope
import pixelcnn.nn as nn
from utils import plotting
from pixelcnn.nn import adam_updates
import utils.mask as m
flatten = tf.contrib.layers.flatten

kernel_initializer = None # tf.random_normal_initializer()

def generative_network(z):
    with tf.variable_scope("generative_network"):
        net = tf.reshape(z, [-1, 1, 1, args.z_dim])

        net = tf.layers.conv2d_transpose(net, 512, 4, strides=1, padding='VALID', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 4x4
        net = tf.layers.conv2d_transpose(net, 256, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 8x8
        net = tf.layers.conv2d_transpose(net, 128, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 16x16
        net = tf.layers.conv2d_transpose(net, 64, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 32x32
        net = tf.layers.conv2d_transpose(net, 32, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 64x64
        net = tf.layers.conv2d_transpose(net, 3, 1, strides=1, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.nn.sigmoid(net)

    return net


def inference_network(x):
    with tf.variable_scope("inference_network"):
        net = tf.reshape(x, [-1, 64, 64, 3]) # 64x64
        net = tf.layers.conv2d(net, 32, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 32x32
        net = tf.layers.conv2d(net, 64, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 16x16
        net = tf.layers.conv2d(net, 128, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 8x8
        net = tf.layers.conv2d(net, 256, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 4x4
        net = tf.layers.conv2d(net, 512, 4, strides=1, padding='VALID', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 1x1
        net = tf.reshape(net, [-1, 512])
        net = tf.layers.dense(net, args.z_dim * 2, activation=None, kernel_initializer=kernel_initializer)
        loc = net[:, :args.z_dim]
        log_var = net[:, args.z_dim:]
    return loc, log_var


def sample_z(loc, log_var):
    with tf.variable_scope("sample_z"):
        scale = tf.sqrt(tf.exp(log_var))
        dist = tf.distributions.Normal()
        z = dist.sample(sample_shape=(args.batch_size, args.z_dim), seed=None)
        z = loc + z * scale
        return z

def vae_model(x):
    loc, log_var = inference_network(x)
    z = sample_z(loc, log_var)
    x_hat = generative_network(z)
    return loc, log_var, z, x_hat


xs = [tf.placeholder(tf.float32, shape=(None, args.image_size, args.image_size, 3)) for i in range(args.nr_gpu)]
ms = [tf.placeholder_with_default(np.ones((args.batch_size, args.image_size, args.image_size), dtype=np.float32), shape=(None, args.image_size, args.image_size)) for i in range(args.nr_gpu)]
mxs = [tf.multiple(xs[i], tf.stack([ms for k in range(3)], axis=-1)) for i in range(args.nr_gpu)]
zs = [tf.placeholder(tf.float32, shape=(None, args.z_dim)) for i in range(args.nr_gpu)]

with tf.variable_scope("vae"):
    for i in range(args.nr_gpu):
        with tf.device('/gpu:%d' % i):
            locs[i], log_vars[i] = inference_network(mxs[i])
            x_hats[i] = generative_network(zs[i])


saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vae'))

def make_feed_dict(data, mgen=None):
    data = np.cast[np.float32](data/255.)
    ds = np.split(data, args.nr_gpu)
    for i in range(args.nr_gpu):
        feed_dict = { xs[i]:ds[i] for i in range(args.nr_gpu) }
    if mgen is not None:
        masks = mgen.gen(data.shape[0])
        masks = np.split(masks, args.nr_gpu)
        for i in range(args.nr_gpu):
            feed_dict.update({ ms[i]:masks[i] for i in range(args.nr_gpu) })
    return feed_dict

def load_vae(sess, saver):

    ckpt_file = FLAGS.save_dir + '/params_' + 'celeba' + '.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, ckpt_file)
