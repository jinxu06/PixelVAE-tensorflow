import os

import sys
import json
import argparse
import time

import numpy as np
import tensorflow as tf

from vae.lvae import VLadderAE


x = tf.placeholder(tf.float32, shape=(8, 64, 64, 3))
vladder = VLadderAE(x, z_dims=None, num_filters=None, beta=1.0)



config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    sess.run(initializer)
