import sys
import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope

def int_shape(x):
    return list(map(int, x.get_shape()))

def log_sum_exp(x, axis):
    return tf.reduce_logsumexp(x, axis=axis)

def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis, keep_dims=True)
    return x - m - tf.log(tf.reduce_sum(tf.exp(x-m), axis, keep_dims=True))

def get_name(layer_name, counters):
    ''' utlity for keeping track of layer names '''
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name

def visualize_samples(images, name="results/test.png", layout=[5,5], vrange=[-1., 1.]):
    images = (images - vrange[0]) / (vrange[1]-vrange[0]) * 255.
    images = np.rint(images).astype(np.uint8)
    view = uf.tile_images(images, size=layout)
    if name is None:
        return view
    view = Image.fromarray(view, 'RGB')
    view.save(name)

def broadcast_masks_tf(masks, num_channels=None, batch_size=None):
    if num_channels is not None:
        masks = tf.stack([masks for i in range(num_channels)], axis=-1)
    if batch_size is not None:
        masks = tf.stack([masks for i in range(batch_size)], axis=0)
    return masks


def get_trainable_variables(flist, filter_type="in"):
    all_vs = tf.trainable_variables()
    if filter_type=="in":
        vs = []
        for s in flist:
            vs += [p for p in all_vs if s in p.name]
    elif filter_type=="not in":
        vs = all_vs
        for s in flist:
            vs = [p for p in vs if s not in p.name]
    return vs


class Recorder(object):

    def __init__(self, dict={}):
        self.dict = dict
        self.keys = self.dict.keys()
        self.fetches = self.__fetches(self.keys)
        self.cur_values = []
        self.epoch_values = []
        self.past_epoch_stats = []
        self.num_epoches = 0

    def __fetches(self, keys):
        fetches = []
        for key in keys:
            fetches.append(self.dict[key])
        return fetches

    def evaluate(self, sess, feed_dict):
        self.cur_values = sess.run(self.fetches, feed_dict=feed_dict)
        self.epoch_values.append(self.cur_values)

    def finish_epoch_and_display(self, keys=None):
        epoch_values = np.array(self.epoch_values)
        stats = np.mean(epoch_values, axis=0)
        self.past_epoch_stats.append(stats)
        self.__display(stats, keys)
        self.epoch_values = []
        self.num_epoches += 1

    def __display(self, stats, keys=None):
        if keys is None:
            keys = self.keys
        results = []
        for k, s in zip(self.keys, stats):
            results[k] = s
        ret_str = "* epoch {0} -- ".format(self.num_epoches)
        for key in keys:
            ret_str += "{0}:{1}   ".format(key, results[key])
        print(ret_str)
        sys.stdout.flush()

    def save(self, keys=None):
        if keys is None:
            keys = self.keys
