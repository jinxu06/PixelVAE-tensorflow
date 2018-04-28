import os
import sys
import json
import argparse
import time
import numpy as np
import tensorflow as tf
import data.load_data as load_data
from blocks.helpers import visualize_samples

# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()
#
# print("TensorFlow version: {}".format(tf.VERSION))
# print("Eager execution: {}".format(tf.executing_eagerly()))

data_set = load_data.CelebA(data_dir="/data/ziz/not-backed-up/jxu/CelebA", batch_size=100, img_size=128)
test_data = data_set.test(shuffle=False)

data = test_data.next(256)
data = np.cast[np.float32]((data - 127.5) / 127.5)

image = data[16*9:16*9+1]
corrupted_image = image.copy()
corrupted_image[:, 50:100, 40:80, :] = 0.
visualize_samples(image, os.path.join("/data/ziz/jxu/gpu-results", "image_example.png"), layout=(1, 1))
visualize_samples(corrupted_image, os.path.join("/data/ziz/jxu/gpu-results", "corrupted_image_example.png"), layout=(1, 1))
