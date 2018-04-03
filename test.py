import numpy as np
import os
import time
import tensorflow as tf

from vae.lvae import VLadderAE


x = tf.placeholder(tf.float32, shape=(8, 64, 64, 3))
vladder = VLadderAE(x, z_dims=None, num_filters=None, beta=1.0)
