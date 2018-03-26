import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope
import pixelcnn.nn as nn

from vae.vae import vae_model
from pixelcnn.mini_model import shallow_pixel_cnn


def pixel_vae(x, f=None, z_dim=32, img_size=64, nr_final_feature_maps=32, dropout_p=0.5, nr_resnet=5, nr_filters=160, nr_logistic_mix=10):

    loc, log_var, z, vae_features = vae_model(x, z_dim=z_dim, img_size=img_size,
                    output_feature_maps=True, nr_final_feature_maps=nr_final_feature_maps)
    if f is None:
        f = vae_features
    x_out = shallow_pixel_cnn(x, sh=f, dropout_p=dropout_p, nr_resnet=nr_resnet,
                    nr_filters=nr_filters, nr_logistic_mix=nr_logistic_mix)

    return x_out, loc, log_var, vae_features, z
