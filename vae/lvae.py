import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
flatten = tf.contrib.layers.flatten

class VLadderAE(object):

    def __init__(self, z_dims=None, num_filters=None, beta=1., reg_type='mmd', counters={}):
        if z_dims is None:
            z_dims = [20, 20, 20]
        if num_filters is None:
            num_filters = [32, 64, 128]
        self.z_dims = z_dims
        self.num_filters = num_filters
        assert len(self.z_dims)==len(self.num_filters), "lengths of z_dims, num_filters do not match"
        self.num_blocks = len(self.z_dims)
        self.beta = beta
        self.reg_type = reg_type
        self.zs = []
        self.z_locs = []
        self.z_scales = []
        self.z_tildes = []
        self.hs = []

        self.counters = counters
        self.nonlinearity = tf.nn.elu

    def build_graph(self, x, mode='train'):
        assert mode in ['train', 'test'], "mode is either train or test"
        print("build graph mode: {0}".format(mode))
        self.__model(x, mode)
        if mode=='train':
            self.__loss(reg=self.reg_type)

    def __model(self, x, mode='train'):
        print("******   Building Graph   ******")
        self.x = x
        h = self.x
        with arg_scope([inference_block, generative_block, ladder_block], nonlinearity=self.nonlinearity, bn=True, counters=self.counters):
            for l in range(self.num_blocks):
                h = inference_block(h, num_filters=self.num_filters[l])
                self.hs.append(h)
                z_loc, z_scale = ladder_block(h, ladder_dim=self.z_dims[l], num_filters=self.num_filters[l])
                self.z_locs.append(z_loc)
                self.z_scales.append(z_scale)
                if mode=='train':
                    z = z_sampler(z_loc, z_scale, counters=self.counters)
                elif mode=='test':
                    z = tf.placeholder(tf.float32, shape=int_shape(z_loc))
                self.zs.append(z)
            z_tilde = None
            for l in reversed(range(self.num_blocks)):
                z_tilde = generative_block(z_tilde, self.zs[l], self.num_filters[l], output_shape=int_shape(self.hs[l])[1:])
                self.z_tildes.append(z_tilde)
            self.x_hat = generative_block(z_tilde, None, 3, nonlinearity=tf.nn.tanh, bn=False)


    def __loss(self, reg='kld'): # reg = kld or mmd or None
        print("******   Compute Loss   ******")
        # self.loss_ae = tf.reduce_mean(tf.reduce_sum(tf.square(flatten(self.x)-flatten(self.x_hat)), 1))
        self.loss_ae = tf.reduce_mean(tf.abs(self.x - self.x_hat))
        if reg is None:
            self.loss = self.loss_ae
            return

        z = tf.concat(self.zs, axis=-1)
        z_loc = tf.concat(self.z_locs, axis=-1)
        z_scale = tf.concat(self.z_scales, axis=-1)
        if reg=='kld':
            z_log_var = tf.log(tf.square(z_scale))
            self.loss_reg = tf.reduce_mean(- 0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_loc) - tf.exp(z_log_var), axis=-1))
        elif reg=='mmd':
            self.loss_reg = compute_mmd(tf.random_normal(int_shape(z)), z)
        print("beta:{0}, reg_type:{1}".format(self.beta, self.reg_type))
        self.loss_ae *= 100
        self.loss_reg *= 100
        self.loss = self.loss_ae + self.beta * self.loss_reg

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
