import numpy as np
import os
import time
import tensorflow as tf

from vae.conv_vae import ConvVAE


x = tf.placeholder(tf.float32, shape=(8, 32, 32, 3))
is_training = tf.placeholder(tf.bool, shape=())
model_opt = {
    "z_dim": 10,
    "reg": "mmd",
    "nonlinearity": tf.nn.elu,
    "bn": True,
    "kernel_initializer": None,
    "kernel_regularizer": None,
}


v = ConvVAE(counters={})
v.build_graph(x, is_training, **model_opt)
