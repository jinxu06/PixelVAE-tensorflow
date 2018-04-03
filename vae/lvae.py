import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
flatten = tf.contrib.layers.flatten

class VLadderAE(object):

    def __init__(self, x, num_blocks, z_dims=None, num_filters=None, counters={}):
        self.num_blocks = num_blocks
        if z_dims is None:
            self.z_dims = [20, 20, 20]
        if num_filters is None:
            self.num_filters = [64, 128, 256]
        self.zs = []
        self.z_locs = []
        self.z_scales = []
        self.z_tildes = []
        self.hs = []
        self.x = x

        self.counters = counters
        self.nonlinearity = tf.nn.elu

        self._build_graph()

    def _build_graph(self):
        print("**** Building Graph ****")
        h = self.x
        with arg_scope([inference_block, generative_block, ladder_block, z_sampler], counters=self.counters):
            for l in range(self.num_blocks):
                h = inference_block(h, num_filters=self.num_filters[l])
                self.hs.append(h)
                z_loc, z_scale = ladder_block(h, ladder_dim=self.z_dims[l], num_filters=self.num_filters[l])
                self.z_locs.append(z_loc)
                self.z_scales.append(z_scale)
                z = z_sampler(z_loc, z_scale)
                self.zs.append(z)
            z_tilde = None
            for l in reversed(range(self.num_blocks)):
                z_tilde = generative_block(z_tilde, self.zs[l], self.num_filters[l], output_shape=int_shape(self.hs[l])[1:])
                self.z_tildes.append(z_tilde)
            self.x_hat = generative_block(z_tilde, None, 3)


    def loss(self, reg='elbo'): # reg = kld or mmd
        pass

@add_arg_scope
def conv2d_layer(inputs, num_filters, kernel_size, strides=1, padding='SAME', nonlinearity=None, bn=True):
    outputs = tf.layers.conv2d(inputs, num_filters, kernel_size=kernel_size, strides=strides, padding=padding)
    if bn:
        outputs = tf.layers.batch_normalization(outputs)
    if nonlinearity is not None:
        outputs = nonlinearity(outputs)
    print("    + conv2d_layer", int_shape(inputs), int_shape(outputs))
    return outputs

@add_arg_scope
def deconv2d_layer(inputs, num_filters, kernel_size, strides=1, padding='SAME', nonlinearity=None, bn=True):
    outputs = tf.layers.conv2d_transpose(inputs, num_filters, kernel_size=kernel_size, strides=strides, padding=padding)
    if bn:
        outputs = tf.layers.batch_normalization(outputs)
    if nonlinearity is not None:
        outputs = nonlinearity(outputs)
    print("    + deconv2d_layer", int_shape(inputs), int_shape(outputs))
    return outputs

@add_arg_scope
def dense_layer(inputs, num_outputs, nonlinearity=None, bn=True):
    inputs_shape = int_shape(inputs)
    assert len(inputs_shape)==2, "inputs should be flattened first"
    outputs = tf.layers.dense(inputs, num_outputs)
    if bn:
        outputs = tf.layers.batch_normalization(outputs)
    if nonlinearity is not None:
        outputs = nonlinearity(outputs)
    print("    + dense_layer", int_shape(inputs), int_shape(outputs))
    return outputs


@add_arg_scope
def z_sampler(loc, scale, counters={}):
    name = get_name("z_sampler", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        dist = tf.distributions.Normal(loc=0., scale=1.)
        z = dist.sample(sample_shape=int_shape(loc), seed=None)
        z = loc + tf.multiply(z, scale)
        print("    + normal_sampler", int_shape(z))
        return z


@add_arg_scope
def generative_block(latent, ladder, num_filters, kernel_size=4, output_shape=None, nonlinearity=None, bn=True, counters={}):
    name = get_name("generative_block", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        if latent is None:
            outputs = combine_noise(latent, ladder, output_shape)
            with arg_scope([deconv2d_layer], kernel_size=kernel_size, nonlinearity=nonlinearity, bn=bn):
                outputs = deconv2d_layer(outputs, num_filters, strides=1)
            return outputs
        outputs= combine_noise(latent, ladder)
        with arg_scope([deconv2d_layer], kernel_size=kernel_size, nonlinearity=nonlinearity, bn=bn):
            outputs = deconv2d_layer(outputs, num_filters, strides=2)
            outputs = deconv2d_layer(outputs, num_filters, strides=1)
            return outputs

@add_arg_scope
def inference_block(inputs, num_filters, kernel_size=4, nonlinearity=None, bn=True, counters={}):
    name = get_name("inference_block", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([conv2d_layer], kernel_size=kernel_size, nonlinearity=nonlinearity, bn=bn):
            outputs = conv2d_layer(inputs, num_filters, strides=1)
            outputs = conv2d_layer(outputs, num_filters, strides=2)
            return outputs

@add_arg_scope
def ladder_block(inputs, ladder_dim, num_filters, kernel_size=4, nonlinearity=None, bn=True, counters={}):
    name = get_name("ladder_block", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([conv2d_layer], kernel_size=kernel_size, nonlinearity=nonlinearity, bn=bn):
            outputs = conv2d_layer(inputs, num_filters, strides=2)
            outputs = conv2d_layer(outputs, num_filters, strides=2)
        outputs = flatten(outputs)
        loc = dense_layer(outputs, ladder_dim, nonlinearity=None)
        scale = dense_layer(outputs, ladder_dim, nonlinearity=tf.sigmoid)
        return loc, scale


def int_shape(x):
    return list(map(int, x.get_shape()))

def get_name(layer_name, counters):
    ''' utlity for keeping track of layer names '''
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name

def combine_noise(latent, ladder, latent_shape=None):
    if latent is None:
        ladder = dense_layer(ladder, np.prod(latent_shape), nonlinearity=tf.nn.elu, bn=True)
        ladder = dense_layer(ladder, np.prod(latent_shape), nonlinearity=tf.nn.elu, bn=True)
        ladder = tf.reshape(ladder, [-1]+latent_shape)
        return ladder
    if ladder is None:
        return latent
    latent_shape = int_shape(latent)[1:]
    ladder = dense_layer(ladder, np.prod(latent_shape), nonlinearity=tf.nn.elu, bn=True)
    ladder = tf.reshape(ladder, [-1]+latent_shape)
    return latent + ladder
    # return tf.concat([latent, ladder], axis=-1)
