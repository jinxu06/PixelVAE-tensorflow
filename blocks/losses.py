import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope


@add_arg_scope
def mix_logistic_loss(x, params, masks=None):
    l = nn.discretized_mix_logistic_loss(x, params, sum_all=False, masks=masks)
    return tf.reduce_mean(l)
