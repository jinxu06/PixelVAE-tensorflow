"""
Trains a Pixel-CNN++ generative model on CIFAR-10 or Tiny ImageNet data.
Uses multiple GPUs, indicated by the flag --nr_gpu

Example usage:
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_double_cnn.py --nr_gpu 4
"""

import os

import sys
import json
import argparse
import time

import numpy as np
import tensorflow as tf

from pixelcnn import nn
from pixelcnn.mini_model import model_spec
from utils import plotting
import utils.mfunc as uf
import utils.mask as um

import vae.load_vae as lv

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    lv.load_vae(sess, lv.saver)
