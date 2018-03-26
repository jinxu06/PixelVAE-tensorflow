import os

import sys
import json
import argparse
import time

import numpy as np
import tensorflow as tf

from pixelcnn import nn
from pixelvae import pixel_vae
from pixelcnn.nn import adam_updates

from utils import plotting
import utils.mfunc as uf
import utils.mask as um
# self define modules
from config import get_config

cfg = get_config('cifar')


parser = argparse.ArgumentParser()


# vae
parser.add_argument('-is', '--img_size', type=int, default=cfg['img_size'], help="size of input image")
parser.add_argument('-zd', '--z_dim', type=int, default=cfg['z_dim'], help="dimension of the latent variable z")
parser.add_argument('-l', '--lam', type=float, default=cfg['lam'], help="threshold under which the KL divergence will not be punished")
parser.add_argument('-b', '--beta', type=float, default=cfg['beta'], help="strength of the KL divergence penalty")
parser.add_argument('-nffm', '--nr_final_feature_maps', type=int, default=cfg['nr_final_feature_maps'], help="number of final feature maps in the vae part")

# data I/O
parser.add_argument('-dd', '--data_dir', type=str, default=cfg['data_dir'], help='Location for the dataset')
parser.add_argument('-sd', '--save_dir', type=str, default=cfg['save_dir'], help='Location for parameter checkpoints and samples')
parser.add_argument('-ds', '--data_set', type=str, default=cfg['data_set'], help='Can be either cifar|imagenet')
parser.add_argument('-si', '--save_interval', type=int, default=cfg['save_interval'], help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-lp', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
# model
parser.add_argument('-nr', '--nr_resnet', type=int, default=cfg['nr_resnet'], help='Number of residual blocks per stage of the model')
parser.add_argument('-nf', '--nr_filters', type=int, default=cfg['nr_filters'], help='Number of filters to use across the model. Higher = larger model.')
parser.add_argument('-nlm', '--nr_logistic_mix', type=int, default=cfg['nr_logistic_mix'], help='Number of logistic components in the mixture. Higher = more flexible model')
# optimization
parser.add_argument('-lr', '--learning_rate', type=float, default=cfg['learning_rate'], help='Base learning rate')
parser.add_argument('-ld', '--lr_decay', type=float, default=cfg['lr_decay'], help='Learning rate decay, applied every step of the optimization')
parser.add_argument('-bs', '--batch_size', type=int, default=cfg['batch_size'], help='Batch size during training per GPU')
parser.add_argument('-ibs', '--init_batch_size', type=int, default=cfg['init_batch_size'], help='How much data to use for data-dependent initialization.')
parser.add_argument('-dp', '--dropout_p', type=float, default=cfg['dropout_p'], help='Dropout strength (i.e. 1 - keep_prob). 0 = No dropout, higher = more dropout.')
parser.add_argument('-me', '--max_epochs', type=int, default=cfg['max_epochs'], help='How many epochs to run in total?')
parser.add_argument('-ng', '--nr_gpu', type=int, default=cfg['nr_gpu'], help='How many GPUs to distribute the training across?')
# reproducibility
parser.add_argument('-s', '--seed', type=int, default=cfg['seed'], help='Random seed to use')
# new features
parser.add_argument('-d', '--debug', dest='debug', action='store_true', help='Under debug mode?')
#
args = parser.parse_args()

print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args

rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)
# initialize data loaders for train/test splits
if args.data_set == 'cifar':
    import data.cifar10_data as cifar10_data
    DataLoader = cifar10_data.DataLoader
elif args.data_set == 'imagenet':
    import data.imagenet_data as imagenet_data
    DataLoader = imagenet_data.DataLoader
elif 'celeba' in args.data_set:
    import data.celeba_data as celeba_data
    DataLoader = celeba_data.DataLoader
else:
    raise("unsupported dataset")
if args.data_set=='celeba128':
    if args.debug:
        train_data = DataLoader(args.data_dir, 'valid', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, size=128)
    else:
        train_data = DataLoader(args.data_dir, 'train', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, size=128)
    test_data = DataLoader(args.data_dir, 'test', args.batch_size * args.nr_gpu, shuffle=False, size=128)
else:
    if args.debug:
        train_data = DataLoader(args.data_dir, 'valid', args.batch_size * args.nr_gpu, rng=rng, shuffle=True)
    else:
        train_data = DataLoader(args.data_dir, 'train', args.batch_size * args.nr_gpu, rng=rng, shuffle=True)
    test_data = DataLoader(args.data_dir, 'test', args.batch_size * args.nr_gpu, shuffle=False)
obs_shape = train_data.get_observation_size() # e.g. a tuple (32,32,3)

# data place holder
xs = [tf.placeholder(tf.float32, shape=(args.batch_size, args.img_size, args.img_size, 3)) for i in range(args.nr_gpu)]
ms = [tf.placeholder_with_default(np.ones((args.batch_size, args.img_size, args.img_size), dtype=np.float32), shape=(args.batch_size, args.img_size, args.img_size)) for i in range(args.nr_gpu)]
mxs = [tf.multiply(xs[i], tf.stack([ms[i] for k in range(3)], axis=-1)) for i in range(args.nr_gpu)]
# zs = [tf.placeholder(tf.float32, shape=(None, args.z_dim)) for i in range(args.nr_gpu)]

# create the model
model_opt = {"z_dim":args.z_dim, "img_size":args.img_size, "nr_final_feature_maps":args.nr_final_feature_maps, "nr_resnet":args.nr_resnet, "nr_filters":args.nr_filters, "nr_logistic_mix":args.nr_logistic_mix}
model = tf.make_template('pixel_vae', pixel_vae)


locs = [None for i in range(args.nr_gpu)]
log_vars = [None for i in range(args.nr_gpu)]


nlls = [None for i in range(args.nr_gpu)]
klds = [None for i in range(args.nr_gpu)]
losses = [None for i in range(args.nr_gpu)]
grads = [None for i in range(args.nr_gpu)]

test_locs = [None for i in range(args.nr_gpu)]
test_log_vars = [None for i in range(args.nr_gpu)]
test_fs = [None for i in range(args.nr_gpu)]
test_nlls = [None for i in range(args.nr_gpu)]
test_klds = [None for i in range(args.nr_gpu)]
test_losses = [None for i in range(args.nr_gpu)]
new_x_gen = [None for i in range(args.nr_gpu)]

for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        out, locs[i], log_vars[i], _, _ = model(mxs[i], dropout_p=args.dropout_p, **model_opt)
        nlls[i] = nn.discretized_mix_logistic_loss(tf.stop_gradient(xs[i]), out, sum_all=False)
        klds[i] = - 0.5 * tf.reduce_mean(1 + log_vars[i] - tf.square(locs[i]) - tf.exp(log_vars[i]), axis=-1)
        losses[i] = nlls[i] + args.beta * tf.maximum(args.lam, klds[i])

        out, test_locs[i], test_log_vars[i], test_fs[i], _ = model(mxs[i], dropout_p=0., **model_opt)
        test_nlls[i] = nn.discretized_mix_logistic_loss(tf.stop_gradient(xs[i]), out, sum_all=False)
        test_klds[i] = - 0.5 * tf.reduce_mean(1 + test_log_vars[i] - tf.square(test_locs[i]) - tf.exp(test_log_vars[i]), axis=-1)
        test_losses[i] = test_nlls[i] + args.beta * tf.maximum(args.lam, test_klds[i])
        epsilon = 0.05

        new_x_gen[i] = nn.sample_from_discretized_mix_logistic(out, args.nr_logistic_mix, epsilon=epsilon)



all_params = tf.trainable_variables()
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        grads[i] = tf.gradients(losses[i], all_params, colocate_gradients_with_ops=True)

with tf.device('/gpu:0'):
    for i in range(1, args.nr_gpu):
        for j in range(len(grads[0])):
            grads[0][j] += grads[i][j]
    nll = tf.concat(nlls, axis=0)
    kld = tf.concat(klds, axis=0)
    loss = tf.concat(losses, axis=0)

    test_nll = tf.concat(test_nlls, axis=0)
    test_kld = tf.concat(test_klds, axis=0)
    test_loss = tf.concat(test_losses, axis=0)

    train_step = adam_updates(all_params, grads[0], lr=args.learning_rate)

initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

def make_feed_dict(data):
    data = np.cast[np.float32]((data - 127.5) / 127.5)
    ds = np.split(data, args.nr_gpu)
    for i in range(args.nr_gpu):
        feed_dict = { xs[i]:ds[i] for i in range(args.nr_gpu) }
    return feed_dict

def sample_from_model(sess, data=None):
    data = np.cast[np.float32]((data - 127.5) / 127.5) ## preprocessing
    ds = np.split(data, args.nr_gpu)

    feed_dict = {xs[i]: ds[i] for i in range(args.nr_gpu)}
    fs_np = sess.run(test_fs, feed_dict=feed_dict)

    handle = sess.partial_run_setup(new_x_gen, test_fs)
    feed_dict = {test_fs[i]: fs_np[i] for i in range(args.nr_gpu)}
    new_x_gen_np = sess.partial_run(handle, new_x_gen, feed_dict=feed_dict)
    print(new_x_gen_np)

    # x_gen = [np.zeros_like(x[0]) for i in range(args.nr_gpu)]
    #
    # for yi in range(obs_shape[0]):
    #     for xi in range(obs_shape[1]):
    #         feed_dict.update({xs[i]: x_gen[i] for i in range(args.nr_gpu)})
    #         new_x_gen_np = sess.run(new_x_gen, feed_dict=feed_dict)
    #         for i in range(args.nr_gpu):
    #             x_gen[i][:,yi,xi,:] = new_x_gen_np[i][:,yi,xi,:]
    # return np.concatenate(x_gen, axis=0)


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
        loss_arr, nll_arr, kld_arr = [], [], []
        for data in train_data:
            feed_dict = make_feed_dict(data)
            l, n, k, _ = sess.run([loss, nll, kld, train_step], feed_dict=feed_dict)
            loss_arr.append(l)
            nll_arr.append(n)
            kld_arr.append(k)
        train_loss, train_nll, train_kld = np.mean(loss_arr), np.mean(nll_arr), np.mean(kld_arr)

        loss_arr, nll_arr, kld_arr = [], [], []
        for data in test_data:
            feed_dict = make_feed_dict(data)
            l, n, k = sess.run([loss, nll, kld], feed_dict=feed_dict)
            loss_arr.append(l)
            nll_arr.append(n)
            kld_arr.append(k)
        test_loss, test_nll, test_kld = np.mean(loss_arr), np.mean(nll_arr), np.mean(kld_arr)

        print("epoch {0} --------------------- Time {1:.2f}s".format(epoch, time.time()-tt))
        print("train loss:{0:.3f}, train nll:{1:.3f}, train kld:{2:.3f}".format(train_loss, train_nll, train_kld))
        print("test loss:{0:.3f}, test nll:{1:.3f}, test kld:{2:.3f}".format(test_loss, test_nll, test_kld))
        sys.stdout.flush()

        if epoch % args.save_interval == 0:

            data = next(test_data)
            sample_x = sample_from_model(sess, data)
            test_data.reset()

        # if epoch % args.save_interval == 0:
        #
        #     saver.save(sess, args.save_dir + '/params_' + 'celeba' + '.ckpt')
        #
        #     data = next(test_data)
        #     sample_x = sample_from_model(sess, data)
        #     test_data.reset()
        #
        #     img_tile = plotting.img_tile(sample_x[:25], aspect_ratio=1.0, border_color=1.0, stretch=True)
        #     img = plotting.plot_img(img_tile, title=args.data_set + ' samples')
        #     plotting.plt.savefig(os.path.join(args.save_dir,'%s_pixelvae_sample%d.png' % (args.data_set, epoch)))
