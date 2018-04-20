import numpy as np
import os
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from blocks.layers import conv2d, deconv2d, dense
from blocks.samplers import gaussian_sampler
from blocks.estimators import estimate_tc, estimate_dwkld, estimate_mi, estimate_mmd
from blocks.losses import gaussian_recons_loss
from blocks.helpers import int_shape
from blocks.components import conv_encoder_64_medium, conv_decoder_64_medium, conv_encoder_32_medium, conv_decoder_32_medium


class ConvVAE(object):

    def __init__(self, counters={}):
        self.counters = counters

    def build_graph(self, x, is_training, z_dim, use_mode="test", reg='mmd', N=2e5, beta=1., lam=0., nonlinearity=tf.nn.elu, bn=True, kernel_initializer=None, kernel_regularizer=None):
        self.z_dim = z_dim
        self.use_mode = use_mode
        self.nonlinearity = nonlinearity
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bn = bn
        self.reg = reg
        self.N = N
        self.beta = beta
        self.lam = lam
        self.__model(x, is_training)
        self.__loss(self.reg)

    def __model(self, x, is_training):
        print("******   Building Graph   ******")
        self.x = x
        self.is_training = is_training
        if int_shape(x)[1]==64:
            encoder = conv_encoder_64_medium
            decoder = conv_decoder_64_medium
        elif int_shape(x)[1]==32:
            encoder = conv_encoder_32_medium
            decoder = conv_decoder_32_medium 
        with arg_scope([encoder, decoder], nonlinearity=self.nonlinearity, bn=self.bn, kernel_initializer=self.kernel_initializer, kernel_regularizer=self.kernel_regularizer, is_training=self.is_training, counters=self.counters):
            self.z_mu, self.z_log_sigma_sq = encoder(x, self.z_dim)
            sigma = tf.exp(self.z_log_sigma_sq / 2.)
            if self.use_mode=='train':
                self.z = gaussian_sampler(self.z_mu, sigma)
            elif self.use_mode=='test':
                self.z = tf.placeholder(tf.float32, shape=int_shape(self.z_mu))
            print("use mode:{0}".format(self.use_mode))
            self.x_hat = decoder(self.z)


    def __loss(self, reg):
        print("******   Compute Loss   ******")
        self.loss_ae = gaussian_recons_loss(self.x, self.x_hat)
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
