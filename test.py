import numpy as np
import os
import time
import tensorflow as tf

from divergence import compute_mmd


x = tf.random_normal((100, 10))
y = tf.random_uniform((100, 10))

mmd = compute_mmd(x, y)

initializer = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    print(sess.run(mmd))
