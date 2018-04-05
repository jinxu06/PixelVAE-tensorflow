import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
flatten = tf.contrib.layers.flatten
from layers import conv2d_layer, deconv2d_layer, dense_layer
from layers import int_shape, get_name, compute_mmd

class ConvVAE(object):

    def __init__(self, counters={}):
        self.counters = counters

    def build_graph(self, x, is_training, z_dim, reg='mmd', beta=1., lam=0., nonlinearity=tf.nn.elu, bn=True, kernel_initializer=None, kernel_regularizer=None):
        self.z_dim = z_dim
        self.nonlinearity = nonlinearity
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bn = bn
        self.reg = reg
        self.beta = beta
        self.lam = lam
        self.__model(x, is_training)
        self.__loss(self.reg)

    def __model(self, x, is_training):
        print("******   Building Graph   ******")
        self.x = x
        self.is_training = is_training
        with arg_scope([conv_encoder_64_block, conv_decoder_64_block], nonlinearity=self.nonlinearity, bn=True, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer, is_training=self.is_training, counters=self.counters):
            self.z_mu, self.z_log_sigma_sq = conv_encoder_64_block(x, self.z_dim)
            sigma = tf.sqrt(tf.exp(self.z_log_sigma_sq))
            self.z = z_sampler(self.z_mu, sigma)
            self.x_hat = conv_decoder_64_block(self.z)


    def __loss(self, reg):
        print("******   Compute Loss   ******")
        # self.loss_ae = tf.reduce_mean(tf.reduce_sum(tf.square(flatten(self.x)-flatten(self.x_hat)), 1))
        self.loss_ae = tf.reduce_mean(tf.square(flatten(self.x)-flatten(self.x_hat)))
        if reg is None:
            self.loss_reg = 0
        elif reg=='kld':
            self.loss_reg = tf.reduce_mean(- 0.5 * tf.reduce_mean(1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq), axis=-1))
        elif reg=='mmd':
            self.loss_reg = compute_mmd(tf.random_normal(int_shape(self.z)), self.z)

        self.loss_ae *= 100
        self.loss_reg *= 100
        print("reg:{0}, beta:{1}, lam:{2}".format(self.reg, self.beta, self.lam))
        self.loss = self.loss_ae + self.beta * tf.maximum(self.lam, self.loss_reg)


@add_arg_scope
def conv_encoder_64_block(inputs, z_dim, is_training, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
    name = get_name("conv_encoder_64_block", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([conv2d_layer, dense_layer], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
            outputs = inputs
            outputs = conv2d_layer(outputs, 32, 1, 1, "SAME")
            outputs = conv2d_layer(outputs, 64, 4, 2, "SAME")
            outputs = conv2d_layer(outputs, 128, 4, 2, "SAME")
            outputs = conv2d_layer(outputs, 256, 4, 2, "SAME")
            outputs = conv2d_layer(outputs, 512, 4, 2, "SAME")
            outputs = conv2d_layer(outputs, 1024, 4, 1, "VALID")
            outputs = tf.reshape(outputs, [-1, 1024])
            z_mu = dense_layer(outputs, z_dim, nonlinearity=None, bn=False)
            z_log_sigma_sq = dense_layer(outputs, z_dim, nonlinearity=None, bn=False)
            return z_mu, z_log_sigma_sq


@add_arg_scope
def conv_decoder_64_block(inputs, is_training, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
    name = get_name("conv_decoder_64_block", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([deconv2d_layer, dense_layer], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
            outputs = dense_layer(inputs, 1024)
            outputs = tf.reshape(outputs, [-1, 1, 1, 1024])
            outputs = deconv2d_layer(outputs, 512, 4, 1, "VALID")
            outputs = deconv2d_layer(outputs, 256, 4, 2, "SAME")
            outputs = deconv2d_layer(outputs, 128, 4, 2, "SAME")
            outputs = deconv2d_layer(outputs, 64, 4, 2, "SAME")
            outputs = deconv2d_layer(outputs, 32, 4, 2, "SAME")
            outputs = deconv2d_layer(outputs, 3, 1, 1, "SAME", nonlinearity=tf.sigmoid, bn=False)
            outputs = 2. * outputs - 1.
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
