import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from blocks.layers import conv2d, deconv2d, dense
from blocks.samplers import gaussian_sampler, mix_logistic_sampler
from blocks.estimators import estimate_tc, estimate_dwkld, estimate_mi, estimate_mmd
from blocks.losses import mix_logistic_loss


class ConvPixelVAE(object):

    def __init__(self, counters={}):
        self.counters = counters

    def build_graph(self, x, x_bar, is_training, dropout_p, z_dim, masks=None, use_mode="test", reg='mmd', N=2e5, sample_range=1.0, beta=1., lam=0., nonlinearity=tf.nn.elu, bn=True, kernel_initializer=None, kernel_regularizer=None, nr_resnet=1, nr_filters=100, nr_logistic_mix=10):
        self.z_dim = z_dim
        self.use_mode = use_mode
        self.nonlinearity = nonlinearity
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bn = bn
        self.reg = reg
        self.N = N
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
                self.z = gaussian_sampler(self.z_mu, sigma)
            elif self.use_mode=='test':
                self.z = tf.placeholder(tf.float32, shape=int_shape(self.z_mu))
            print("use mode:{0}".format(self.use_mode))
            self.decoded_features = deconv_block(self.z)
            if self.masks is None:
                sh = self.decoded_features
            else:
                self.encoded_context = encode_context_block(self.x, self.masks)
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
            self.loss_reg = estimate_mmd(tf.random_normal(int_shape(self.z)), self.z)
            self.loss_reg = self.beta * tf.maximum(self.lam, self.loss_reg)
        elif reg=='tc':
            self.mi = estimate_mi(self.z, self.z_mu, self.z_log_sigma_sq, N=self.N)
            self.tc = estimate_tc(self.z, self.z_mu, self.z_log_sigma_sq, N=self.N)
            self.dwkld = estimate_dwkld(self.z, self.z_mu, self.z_log_sigma_sq, N=self.N)
            self.loss_reg = self.mi + self.beta * self.tc + self.dwkld
        print("reg:{0}, beta:{1}, lam:{2}".format(self.reg, self.beta, self.lam))
        self.loss = self.loss_ae + self.loss_reg
