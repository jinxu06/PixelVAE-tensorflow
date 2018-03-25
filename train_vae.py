import numpy as np
import os
import tensorflow as tf
import argparse
import json
import time
import data.celeba_data as celeba_data
from tensorflow.contrib.framework.python.ops import arg_scope
import pixelcnn.nn as nn
from utils import plotting
from pixelcnn.nn import adam_updates
import utils.mask as m
flatten = tf.contrib.layers.flatten


parser = argparse.ArgumentParser()

parser.add_argument('-is', '--image_size', type=int, default=64, help="size of input image")
parser.add_argument('-zd', '--z_dim', type=int, default=32, help="dimension of the latent variable z")
parser.add_argument('-l', '--lam', type=float, default=1., help="threshold under which the KL divergence will not be punished")
parser.add_argument('-b', '--beta', type=float, default=1., help="strength of the KL divergence penalty")

parser.add_argument('-bs', '--batch_size', type=int, default=100, help="batch size")
parser.add_argument('-si', '--save_interval', type=int, default=10, help="epoch interval for checkpointing")

parser.add_argument('-dd', '--data_dir', type=str, default="/data/ziz/not-backed-up/jxu/CelebA", help="data storage location")
parser.add_argument('-sd', '--save_dir', type=str, default="/data/ziz/jxu/models/vae-test", help="checkpoints storage location")
parser.add_argument('-ng', '--nr_gpu', type=int, default=1, help="number of GPUs used")
parser.add_argument('-ds', '--data_set', type=str, default="celeba64", help="dataset used")

parser.add_argument('-lp', '--load_params', dest='load_params', action='store_true', help='whether to load previous checkpoint?')
parser.add_argument('-d', '--debug', dest='debug', action='store_true', help='Under debug mode?')

args = parser.parse_args()

print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args

# Data IO
if args.debug:
    train_data = celeba_data.DataLoader(args.data_dir, 'valid', args.batch_size*args.nr_gpu, shuffle=True, size=args.image_size)
else:
    train_data = celeba_data.DataLoader(args.data_dir, 'train', args.batch_size*args.nr_gpu, shuffle=True, size=args.image_size)
test_data = celeba_data.DataLoader(args.data_dir, 'valid', args.batch_size*args.nr_gpu, shuffle=False, size=args.image_size)


kernel_initializer = None # tf.random_normal_initializer()

def generative_network(z):
    with tf.variable_scope("generative_network"):
        net = tf.reshape(z, [-1, 1, 1, args.z_dim])

        net = tf.layers.conv2d_transpose(net, 512, 4, strides=1, padding='VALID', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 4x4
        net = tf.layers.conv2d_transpose(net, 256, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 8x8
        net = tf.layers.conv2d_transpose(net, 128, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 16x16
        net = tf.layers.conv2d_transpose(net, 64, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 32x32
        net = tf.layers.conv2d_transpose(net, 32, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 64x64
        net = tf.layers.conv2d_transpose(net, 3, 1, strides=1, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.nn.sigmoid(net)

    return net


def inference_network(x):
    with tf.variable_scope("inference_network"):
        net = tf.reshape(x, [-1, 64, 64, 3]) # 64x64
        net = tf.layers.conv2d(net, 32, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 32x32
        net = tf.layers.conv2d(net, 64, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 16x16
        net = tf.layers.conv2d(net, 128, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 8x8
        net = tf.layers.conv2d(net, 256, 5, strides=2, padding='SAME', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 4x4
        net = tf.layers.conv2d(net, 512, 4, strides=1, padding='VALID', kernel_initializer=kernel_initializer)
        net = tf.layers.batch_normalization(net)
        net = tf.nn.elu(net) # 1x1
        net = tf.reshape(net, [-1, 512])
        net = tf.layers.dense(net, args.z_dim * 2, activation=None, kernel_initializer=kernel_initializer)
        loc = net[:, :args.z_dim]
        log_var = net[:, args.z_dim:]
    return loc, log_var


def sample_z(loc, log_var):
    with tf.variable_scope("sample_z"):
        scale = tf.sqrt(tf.exp(log_var))
        dist = tf.distributions.Normal()
        z = dist.sample(sample_shape=(args.batch_size, args.z_dim), seed=None)
        z = loc + tf.multiply(z, scale)
        return z

def vae_model(x):
    loc, log_var = inference_network(x)
    z = sample_z(loc, log_var)
    x_hat = generative_network(z)
    return loc, log_var, z, x_hat


model_opt = {}
model = tf.make_template('vae', vae_model)

xs = [tf.placeholder(tf.float32, shape=(None, args.image_size, args.image_size, 3)) for i in range(args.nr_gpu)]
ms = [tf.placeholder_with_default(np.ones((args.batch_size, args.image_size, args.image_size), dtype=np.float32), shape=(None, args.image_size, args.image_size)) for i in range(args.nr_gpu)]
mxs = [tf.multiply(xs[i], tf.stack([ms for k in range(3)], axis=-1)) for i in range(args.nr_gpu)]


locs = [None for i in range(args.nr_gpu)]
log_vars = [None for i in range(args.nr_gpu)]
zs = [None for i in range(args.nr_gpu)]
x_hats = [None for i in range(args.nr_gpu)]

MSEs = [None for i in range(args.nr_gpu)]
KLDs = [None for i in range(args.nr_gpu)]
losses = [None for i in range(args.nr_gpu)]
grads = [None for i in range(args.nr_gpu)]



for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        locs[i], log_vars[i], zs[i], x_hats[i] = model(mxs[i], **model_opt)

all_params = tf.trainable_variables(scope='vae')
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        MSEs[i] = tf.reduce_sum(tf.square(flatten(xs[i])-flatten(x_hats[i])), 1)
        KLDs[i] = - 0.5 * tf.reduce_mean(1 + log_vars[i] - tf.square(locs[i]) - tf.exp(log_vars[i]), axis=-1)
        losses[i] = tf.reduce_mean(MSEs[i] + args.beta * tf.maximum(args.lam, KLDs[i]))
        grads[i] = tf.gradients(losses[i], all_params, colocate_gradients_with_ops=True)

with tf.device('/gpu:0'):
    for i in range(1, args.nr_gpu):
        losses[0] += losses[i]
        for j in range(len(grads[0])):
            grads[0][j] += grads[i][j]

    MSE = tf.concat(MSEs, axis=0)
    KLD = tf.concat(KLDs, axis=0)

    train_step = adam_updates(all_params, grads[0], lr=0.0001)

    loss = losses[0] / args.nr_gpu

initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

def make_feed_dict(data, mgen=None):
    data = np.cast[np.float32](data/255.)
    ds = np.split(data, args.nr_gpu)
    for i in range(args.nr_gpu):
        feed_dict = { xs[i]:ds[i] for i in range(args.nr_gpu) }
    if mgen is not None:
        masks = mgen.gen(data.shape[0])
        masks = np.split(masks, args.nr_gpu)
        for i in range(args.nr_gpu):
            feed_dict.update({ ms[i]:masks[i] for i in range(args.nr_gpu) })
    return feed_dict


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    # init
    sess.run(initializer)

    if args.load_params:
        ckpt_file = args.save_dir + '/params_' + 'celeba' + '.ckpt'
        print('restoring parameters from', ckpt_file)
        saver.restore(sess, ckpt_file)

    # train_mgen = m.RandomRectangleMaskGenerator(args.image_size, args.image_size, max_ratio=0.75)
    # test_mgen = m.CenterMaskGenerator(args.image_size, args.image_size, 0.5)

    max_num_epoch = 1000
    for epoch in range(max_num_epoch):
        tt = time.time()
        ls, mses, klds = [], [], []
        for data in train_data:
            feed_dict = make_feed_dict(data)
            l, mse, kld, _ = sess.run([loss, MSE, KLD, train_step], feed_dict=feed_dict)
            ls.append(l)
            mses.append(mse)
            klds.append(kld)
        train_loss, train_mse, train_kld = np.mean(ls), np.mean(mses), np.mean(klds)

        ls, mses, klds = [], [], []
        for data in test_data:
            feed_dict = make_feed_dict(data)
            l, mse, kld = sess.run([loss, MSE, KLD], feed_dict=feed_dict)
            ls.append(l)
            mses.append(mse)
            klds.append(kld)
        test_loss, test_mse, test_kld = np.mean(ls), np.mean(mses), np.mean(klds)

        print("epoch {0} --------------------- Time {1:.2f}s".format(epoch, time.time()-tt))
        print("train loss:{0:.3f}, train mse:{1:.3f}, train kld:{2:.3f}".format(train_loss, train_mse, train_kld))
        print("test loss:{0:.3f}, test mse:{1:.3f}, test kld:{2:.3f}".format(test_loss, test_mse, test_kld))

        if epoch % args.save_interval == 0:

            saver.save(sess, args.save_dir + '/params_' + 'celeba' + '.ckpt')

            data = next(test_data)
            feed_dict = make_feed_dict(data)
            sample_x = sess.run(x_hats, feed_dict=feed_dict)
            sample_x = np.concatenate(sample_x, axis=0)
            test_data.reset()

            img_tile = plotting.img_tile(sample_x[:25], aspect_ratio=1.0, border_color=1.0, stretch=True)
            img = plotting.plot_img(img_tile, title=args.data_set + ' samples')
            plotting.plt.savefig(os.path.join(args.save_dir,'%s_vae_sample%d.png' % (args.data_set, epoch)))
