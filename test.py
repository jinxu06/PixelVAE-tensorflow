import numpy as np
import os
import time
import tensorflow as tf

import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))



def estimate_log_probs(z, z_mu, z_log_sigma_sq, N=200000):
    z_b = tf.stack([z for i in range(batch_size)], axis=0)

    z_mu_b = tf.stack([z_mu for i in range(batch_size)], axis=1)
    z_sigma_b = tf.stack([z_sigma for i in range(batch_size)], axis=1)
    z_norm = (z_b-z_mu_b) / z_sigma_b

    dist = tf.distributions.Normal(loc=0., scale=1.)
    log_probs = dist.log_prob(z_norm)
    lse_sum = tf.reduce_mean(log_sum_exp(tf.reduce_sum(log_probs, axis=-1), axis=0))
    sum_lse = tf.reduce_mean(tf.reduce_sum(log_sum_exp(log_probs, axis=0), axis=-1))
    return log_probs



batch_size = 512
z_dim = 16

z_mu = np.random_normal(size=(batch_size, z_dim))
z_log_sigma_sq = np.random_normal(size=(batch_size, z_dim))
z_sigma = np.sqrt(np.exp(z_log_sigma_sq))
z = np.random_normal(loc=z_mu, scale=z_sigma)

e = estimate_log_probs(z, z_mu, z_log_sigma_sq)
print(e)
