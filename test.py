import numpy as np
import os
import time
import tensorflow as tf

from vae.cond_pixel_vae import ConvPixelVAE

x = tf.placeholder(tf.float32, shape=(8, 64, 64, 3))
# sh = tf.placeholder(tf.float32, shape=(8, 64, 64, 32))


x = tf.placeholder(tf.float32, shape=(8, 64, 64, 3))
is_trainings = tf.placeholder(tf.bool, shape=())

pvae = ConvPixelVAE(counters={})
model_opt = {
    "z_dim": 32,
    "reg": "mmd",
}
model = tf.make_template('PVAE', ConvPixelVAE.build_graph)

model(pvae, x,  is_training, **model_opt)
