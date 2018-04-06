import numpy as np
import os
import time
import tensorflow as tf

from cond_pixel_cnn import cond_pixel_cnn

x = tf.placeholder(tf.float32, shape=(8, 64, 64, 3))
sh = tf.placeholder(tf.float32, shape=(8, 64, 64, 32))


out = cond_pixel_cnn(x, sh=sh)
