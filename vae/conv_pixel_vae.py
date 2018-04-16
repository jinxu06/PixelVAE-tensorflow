import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
flatten = tf.contrib.layers.flatten
from layers import conv2d_layer, deconv2d_layer, dense_layer
from layers import compute_mmd, compute_tc, compute_dwkld, compute_entropy, compute_mi
from layers import int_shape, get_name, broadcast_masks_tf
from cond_pixel_cnn import cond_pixel_cnn, mix_logistic_sampler, mix_logistic_loss


class ConvPixelVAE(object):

    def __init__(self, counters={}):
        self.counters = counters

    def build_graph(self, x, x_bar, is_training, dropout_p, z_dim, masks=None, use_mode="test", reg='mmd', sample_range=1.0, beta=1., lam=0., nonlinearity=tf.nn.elu, bn=True, kernel_initializer=None, kernel_regularizer=None, nr_resnet=1, nr_filters=100, nr_logistic_mix=10):
        self.z_dim = z_dim
        self.use_mode = use_mode
        self.nonlinearity = nonlinearity
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bn = bn
        self.reg = reg
        self.sample_range = sample_range
        self.beta = beta
        self.lam = lam
        self.nr_resnet = nr_resnet
        self.nr_filters = nr_filters
        self.nr_logistic_mix = nr_logistic_mix
        self.__model(x, x_bar, is_training, dropout_p, masks)
        self.__loss(self.reg)


    def __model(self, x, x_bar, is_training, dropout_p, masks):
        print("******   Building Graph   ******")
        self.x = x
        self.x_bar = x_bar
        self.is_training = is_training
        self.dropout_p = dropout_p
        self.masks = masks
        if int_shape(x)[1]==64:
            conv_block = conv_encoder_64_block
            deconv_block = deconv_64_block
        elif int_shape(x)[1]==32:
            conv_block = conv_encoder_32_block
            deconv_block = deconv_32_block
        with arg_scope([conv_block, deconv_block, encode_context_block], nonlinearity=self.nonlinearity, bn=self.bn, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer, is_training=self.is_training, counters=self.counters):
            self.z_mu, self.z_log_sigma_sq = conv_block(x, self.z_dim)
            sigma = tf.exp(self.z_log_sigma_sq / 2.)
            if self.use_mode=='train':
                self.z = z_sampler(self.z_mu, sigma)
            elif self.use_mode=='test':
                self.z = tf.placeholder(tf.float32, shape=int_shape(self.z_mu))
            print("use mode:{0}".format(self.use_mode))
            self.decoded_features = deconv_block(self.z)
            if self.masks is None:
                sh = self.decoded_features
            else:
                self.encoded_context = encode_context_block(self.x, self.masks, bn=True)
                sh = tf.concat([self.decoded_features, self.encoded_context], axis=-1)
            self.mix_logistic_params = cond_pixel_cnn(self.x_bar, sh=sh, nonlinearity=self.nonlinearity, nr_resnet=self.nr_resnet, nr_filters=self.nr_filters, nr_logistic_mix=self.nr_logistic_mix, bn=self.bn, dropout_p=self.dropout_p, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer, is_training=self.is_training, counters=self.counters)
            self.x_hat = mix_logistic_sampler(self.mix_logistic_params, nr_logistic_mix=self.nr_logistic_mix, sample_range=self.sample_range, counters=self.counters)


    def __loss(self, reg):
        print("******   Compute Loss   ******")
        self.loss_ae = mix_logistic_loss(self.x, self.mix_logistic_params, masks=self.masks)
        if reg is None:
            self.loss_reg = 0
        elif reg=='kld':
            self.loss_reg = tf.reduce_mean(- 0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq), axis=-1))
            self.loss_reg = self.beta * tf.maximum(self.lam, self.loss_reg)
        elif reg=='mmd':
            self.loss_reg = compute_mmd(tf.random_normal(int_shape(self.z)), self.z)
            self.loss_reg = self.beta * tf.maximum(self.lam, self.loss_reg)
        elif reg=='tc':
            # tc = compute_tc(self.z, self.z_mu, self.z_log_sigma_sq)
            # kld = tf.reduce_mean(- 0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq), axis=-1))
            # self.loss_reg = kld + (self.beta-1.) * tc
            mi = compute_mi(self.z, self.z_mu, self.z_log_sigma_sq)
            tc = compute_tc(self.z, self.z_mu, self.z_log_sigma_sq)
            dwkld = compute_dwkld(self.z, self.z_mu, self.z_log_sigma_sq)
            self.loss_reg = mi + self.beta * tc + dwkld
        elif reg=='tc-dwkld':
            tc = compute_tc(self.z, self.z_mu, self.z_log_sigma_sq)
            dwkld = compute_dwkld(self.z, self.z_mu, self.z_log_sigma_sq)
            self.loss_reg = self.beta * tc + dwkld
        elif reg=='ce-tc-dwkld':
            tc = compute_tc(self.z, self.z_mu, self.z_log_sigma_sq)
            dwkld = compute_dwkld(self.z, self.z_mu, self.z_log_sigma_sq)
            cond_entropy = tf.reduce_mean(compute_entropy(self.z_mu, self.z_log_sigma_sq))
            self.loss_reg = self.beta * tc + dwkld - cond_entropy

        #self.loss_ae *= 100
        #self.loss_reg *= 100
        print("reg:{0}, beta:{1}, lam:{2}".format(self.reg, self.beta, self.lam))
        self.loss = self.loss_ae + self.loss_reg


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
def deconv_64_block(inputs, is_training, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
    name = get_name("deconv_64_block", counters)
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
            return outputs

@add_arg_scope
def conv_encoder_32_block(inputs, z_dim, is_training, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
    name = get_name("conv_encoder_32_block", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([conv2d_layer, dense_layer], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
            outputs = inputs
            outputs = conv2d_layer(outputs, 32, 1, 1, "SAME")
            outputs = conv2d_layer(outputs, 64, 4, 2, "SAME")
            outputs = conv2d_layer(outputs, 128, 4, 2, "SAME")
            outputs = conv2d_layer(outputs, 256, 4, 2, "SAME")
            outputs = conv2d_layer(outputs, 512, 4, 1, "VALID")
            outputs = tf.reshape(outputs, [-1, 512])
            z_mu = dense_layer(outputs, z_dim, nonlinearity=None, bn=True)
            z_log_sigma_sq = dense_layer(outputs, z_dim, nonlinearity=None, bn=True)
            return z_mu, z_log_sigma_sq


@add_arg_scope
def conv_decoder_32_block(inputs, is_training, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
    name = get_name("conv_decoder_32_block", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([deconv2d_layer, dense_layer], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
            outputs = dense_layer(inputs, 512)
            outputs = tf.reshape(outputs, [-1, 1, 1, 512])
            outputs = deconv2d_layer(outputs, 256, 4, 1, "VALID")
            outputs = deconv2d_layer(outputs, 128, 4, 2, "SAME")
            outputs = deconv2d_layer(outputs, 64, 4, 2, "SAME")
            outputs = deconv2d_layer(outputs, 32, 4, 2, "SAME")
            outputs = deconv2d_layer(outputs, 3, 1, 1, "SAME", nonlinearity=tf.sigmoid, bn=True)
            outputs = 2. * outputs - 1.
            return outputs

@add_arg_scope
def deconv_32_block(inputs, is_training, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
    name = get_name("deconv_32_block", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([deconv2d_layer, dense_layer], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
            outputs = dense_layer(inputs, 512)
            outputs = tf.reshape(outputs, [-1, 1, 1, 512])
            outputs = deconv2d_layer(outputs, 256, 4, 1, "VALID")
            outputs = deconv2d_layer(outputs, 128, 4, 2, "SAME")
            outputs = deconv2d_layer(outputs, 64, 4, 2, "SAME")
            outputs = deconv2d_layer(outputs, 32, 4, 2, "SAME")
            return outputs


@add_arg_scope
def encode_context_block(contexts, masks, is_training, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, counters={}):
    name = get_name("encode_context_block", counters)
    print("construct", name, "...")
    with tf.variable_scope(name):
        with arg_scope([conv2d_layer, deconv2d_layer], nonlinearity=nonlinearity, bn=bn, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, is_training=is_training):
            outputs = contexts * broadcast_masks_tf(masks, num_channels=3)
            outputs = tf.concat([outputs, broadcast_masks_tf(masks, num_channels=1)], axis=-1)
            outputs = conv2d_layer(outputs, 32, 3, 1, "SAME")
            outputs = conv2d_layer(outputs, 32, 3, 1, "SAME")
            res1 = outputs
            outputs = conv2d_layer(outputs, 64, 3, 2, "SAME")
            res2 = outputs
            outputs = conv2d_layer(outputs, 128, 3, 2, "SAME")
            res3 = outputs
            outputs = conv2d_layer(outputs, 128, 3, 1, "SAME", nonlinearity=None)
            outputs = nonlinearity(outputs + res3)
            outputs = deconv2d_layer(outputs, 64, 3, 2, "SAME", nonlinearity=None)
            outputs = nonlinearity(outputs + res2)
            outputs = deconv2d_layer(outputs, 32, 3, 2, "SAME", nonlinearity=None)
            outputs = nonlinearity(outputs + res1)
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
