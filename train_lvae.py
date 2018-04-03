import os
import sys
import json
import argparse
import time
from pixelcnn.nn import adam_updates
import numpy as np
import tensorflow as tf
from vae.lvae import VLadderAE

parser = argparse.ArgumentParser()

cfg = {
    "img_size": 32,
    "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
    "save_dir": "/data/ziz/jxu/models/lvae-celeba32",
    "data_set": "celeba32",
    "batch_size": 8,
    "nr_gpu": 1,
}

parser.add_argument('-is', '--img_size', type=int, default=cfg['img_size'], help="size of input image")
# data I/O
parser.add_argument('-dd', '--data_dir', type=str, default=cfg['data_dir'], help='Location for the dataset')
parser.add_argument('-sd', '--save_dir', type=str, default=cfg['save_dir'], help='Location for parameter checkpoints and samples')
parser.add_argument('-ds', '--data_set', type=str, default=cfg['data_set'], help='Can be either cifar|imagenet')
#parser.add_argument('-si', '--save_interval', type=int, default=cfg['save_interval'], help='Every how many epochs to write checkpoint/samples?')
#parser.add_argument('-lp', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
parser.add_argument('-bs', '--batch_size', type=int, default=cfg['batch_size'], help='Batch size during training per GPU')
parser.add_argument('-ng', '--nr_gpu', type=int, default=cfg['nr_gpu'], help='How many GPUs to distribute the training across?')
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
# new features
parser.add_argument('-d', '--debug', dest='debug', action='store_true', help='Under debug mode?')

rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)

import data.celeba_data as celeba_data
DataLoader = celeba_data.DataLoader
if args.debug:
    train_data = DataLoader(args.data_dir, 'valid', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, size=args.img_size)
else:
    train_data = DataLoader(args.data_dir, 'train', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, size=args.img_size)
test_data = DataLoader(args.data_dir, 'test', args.batch_size * args.nr_gpu, shuffle=False, size=args.img_size)


x = tf.placeholder(tf.float32, shape=(args.batch_size, args.img_size, args.img_size, 3))
vladder = VLadderAE(x, z_dims=None, num_filters=None, beta=1.0)


initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    sess.run(initializer)
    print(vladder)
