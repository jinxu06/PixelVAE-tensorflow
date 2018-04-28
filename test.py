import os
import sys
import json
import argparse
import time
import numpy as np
import tensorflow as tf
import data.load_data as load_data

import tensorflow.contrib.eager as tfe
tfe.enable_eager_execution()

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))

data_set = load_data.CelebA(data_dir=args.data_dir, batch_size=batch_size, img_size=args.img_size)
test_data = data_set.test(shuffle=False)

data = test_data.next(256)
data = np.cast[np.float32]((data - 127.5) / 127.5)
visualize_samples(data, os.path.join("results", "examples.png"), layout=(16, 16))
