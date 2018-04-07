import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from PIL import Image
import utils.mfunc as uf

@add_arg_scope
def conv2d_layer(inputs, num_filters, kernel_size, strides=1, padding='SAME', nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False):
    outputs = tf.layers.conv2d(inputs, num_filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    if bn:
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
    if nonlinearity is not None:
        outputs = nonlinearity(outputs)
    print("    + conv2d_layer", int_shape(inputs), int_shape(outputs))
    return outputs

@add_arg_scope
def deconv2d_layer(inputs, num_filters, kernel_size, strides=1, padding='SAME', nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False):
    outputs = tf.layers.conv2d_transpose(inputs, num_filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    if bn:
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
    if nonlinearity is not None:
        outputs = nonlinearity(outputs)
    print("    + deconv2d_layer", int_shape(inputs), int_shape(outputs))
    return outputs

@add_arg_scope
def dense_layer(inputs, num_outputs, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False):
    inputs_shape = int_shape(inputs)
    assert len(inputs_shape)==2, "inputs should be flattened first"
    outputs = tf.layers.dense(inputs, num_outputs, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    if bn:
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
    if nonlinearity is not None:
        outputs = nonlinearity(outputs)
    print("    + dense_layer", int_shape(inputs), int_shape(outputs))
    return outputs




def int_shape(x):
    return list(map(int, x.get_shape()))

def log_sum_exp(x, axis):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    #axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x-m2), axis))

def get_name(layer_name, counters):
    ''' utlity for keeping track of layer names '''
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name


def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

def compute_mmd(x, y, sigma_sqr=1.0):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)
    return mmd

def compute_tc(z, z_mu, z_log_sigma_sq):
    z_sigma = tf.sqrt(tf.exp(z_log_sigma_sq))
    log_probs = []
    batch_size, z_dim = int_shape(z_mu)
    z = tf.stack([z for i in range(batch_size)], axis=0)
    z_mu = tf.stack([z_mu for i in range(batch_size)], axis=1)
    z_sigma = tf.stack([z_sigma for i in range(batch_size)], axis=1)
    z_norm = (z-z_mu) / z_sigma

    dist = tf.distributions.Normal(loc=0., scale=1.)
    log_probs = dist.log_prob(z_norm)
    lse_sum = log_sum_exp(tf.reduce_sum(log_probs, axis=-1), axis=1)
    sum_lse = tf.reduce_sum(log_sum_exp(log_probs, axis=1), axis=-1)
    return lse_sum - sum_lse


def visualize_samples(images, name="results/test.png", layout=[5,5], vrange=[-1, 1]):
    images = (images + vrange[0]) / (vrange[1]-vrange[0]) * 255.
    images = np.rint(images).astype(np.uint8)
    view = Image.fromarray(uf.tile_images(images, size=layout), 'RGB')
    if name is None:
        return view
    view.save(name)
