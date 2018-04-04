import os
import sys
import json
import argparse
import time
from pixelcnn.nn import adam_updates
import numpy as np
import tensorflow as tf
from utils import plotting
from vae.lvae import VLadderAE

parser = argparse.ArgumentParser()

cfg = {
    "img_size": 64,
    "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
    "save_dir": "/data/ziz/jxu/models/lvae-celeba32",
    "data_set": "celeba64",
    "batch_size": 32,
    "nr_gpu": 1,
    "learning_rate": 0.0001,
    "beta": 5e4,
    "save_interval": 10,
}

parser.add_argument('-is', '--img_size', type=int, default=cfg['img_size'], help="size of input image")
# data I/O
parser.add_argument('-dd', '--data_dir', type=str, default=cfg['data_dir'], help='Location for the dataset')
parser.add_argument('-sd', '--save_dir', type=str, default=cfg['save_dir'], help='Location for parameter checkpoints and samples')
parser.add_argument('-ds', '--data_set', type=str, default=cfg['data_set'], help='Can be either cifar|imagenet')
parser.add_argument('-si', '--save_interval', type=int, default=cfg['save_interval'], help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-lp', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
parser.add_argument('-bs', '--batch_size', type=int, default=cfg['batch_size'], help='Batch size during training per GPU')
parser.add_argument('-ng', '--nr_gpu', type=int, default=cfg['nr_gpu'], help='How many GPUs to distribute the training across?')
parser.add_argument('-lr', '--learning_rate', type=float, default=cfg['learning_rate'], help='Base learning rate')
parser.add_argument('-b', '--beta', type=float, default=cfg['beta'], help="strength of the KL divergence penalty")
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
# new features
parser.add_argument('-d', '--debug', dest='debug', action='store_true', help='Under debug mode?')

args = parser.parse_args()
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args

rng = np.random.RandomState(args.seed)
tf.set_random_seed(args.seed)



import data.celeba_data as celeba_data
DataLoader = celeba_data.DataLoader
if args.debug:
    train_data = DataLoader(args.data_dir, 'valid', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, size=args.img_size)
else:
    train_data = DataLoader(args.data_dir, 'train', args.batch_size * args.nr_gpu, rng=rng, shuffle=True, size=args.img_size)
test_data = DataLoader(args.data_dir, 'test', args.batch_size * args.nr_gpu, shuffle=False, size=args.img_size)


xs = [tf.placeholder(tf.float32, shape=(args.batch_size, args.img_size, args.img_size, 3)) for i in range(args.nr_gpu)]

z_dims = [10, 10, 10, 10]
num_filters = [64, 128, 256, 512]
vladders = [VLadderAE(z_dims=z_dims, num_filters=num_filters, beta=args.beta, reg_type="mmd", counters={}) for i in range(args.nr_gpu)]

model_opt = {"mode": 'train'}
model = tf.make_template('build_graph', VLadderAE.build_graph)

for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        model(vladders[i], xs[i])

all_params = tf.trainable_variables()

grads = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        grads.append(tf.gradients(vladders[i].loss, all_params, colocate_gradients_with_ops=True))

with tf.device('/gpu:0'):
    for i in range(1, args.nr_gpu):
        for j in range(len(grads[0])):
            grads[0][j] += grads[i][j]

    loss = tf.add_n([v.loss for v in vladders]) / args.nr_gpu
    loss_ae = tf.add_n([v.loss_ae for v in vladders]) / args.nr_gpu
    loss_reg = tf.add_n([v.loss_reg for v in vladders]) / args.nr_gpu

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = adam_updates(all_params, grads[0], lr=args.learning_rate)



def make_feed_dict(data):
    data = np.cast[np.float32]((data - 127.5) / 127.5)
    ds = np.split(data, args.nr_gpu)
    for i in range(args.nr_gpu):
        feed_dict = { xs[i]:ds[i] for i in range(args.nr_gpu) }
    return feed_dict

def sample_from_model(sess, data):
    data = np.cast[np.float32]((data - 127.5) / 127.5)
    ds = np.split(data, args.nr_gpu)
    feed_dict = { xs[i]:ds[i] for i in range(args.nr_gpu) }
    x_hats = sess.run([vladders[i].x_hat for i in range(args.nr_gpu)], feed_dict=feed_dict)
    return np.concatenate(x_hats, axis=0)




initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    sess.run(initializer)

    if args.load_params:
        ckpt_file = args.save_dir + '/params_' + args.data_set + '.ckpt'
        print('restoring parameters from', ckpt_file)
        saver.restore(sess, ckpt_file)

    max_num_epoch = 200
    for epoch in range(max_num_epoch):
        tt = time.time()
        loss_arr, loss_ae_arr, loss_reg_arr = [], [], []
        for data in train_data:
            feed_dict = make_feed_dict(data)
            _, l, la, lr = sess.run([train_step, loss, loss_ae, loss_reg], feed_dict=feed_dict)
            loss_arr.append(l)
            loss_ae_arr.append(la)
            loss_reg_arr.append(lr)
        train_loss, train_loss_ae, train_loss_reg = np.mean(loss_arr), np.mean(loss_ae_arr), np.mean(loss_reg_arr)

        loss_arr, loss_ae_arr, loss_reg_arr = [], [], []
        for data in test_data:
            feed_dict = make_feed_dict(data)
            l, la, lr = sess.run([loss, loss_ae, loss_reg], feed_dict=feed_dict)
            loss_arr.append(l)
            loss_ae_arr.append(la)
            loss_reg_arr.append(lr)

            print(l, la, lr)
        test_loss, test_loss_ae, test_loss_reg = np.mean(loss_arr), np.mean(loss_ae_arr), np.mean(loss_reg_arr)

        print("epoch {0} --------------------- Time {1:.2f}s".format(epoch, time.time()-tt))
        print("train loss:{0:.3f}, train ae loss:{1:.3f}, train reg loss:{2:.6f}".format(train_loss, train_loss_ae, train_loss_reg))
        print("test loss:{0:.3f}, test ae loss:{1:.3f}, test reg loss:{2:.6f}".format(test_loss, test_loss_ae, test_loss_reg))
        sys.stdout.flush()

        if epoch % args.save_interval == 0:

            saver.save(sess, args.save_dir + '/params_' + args.data_set + '.ckpt')
            data = next(test_data)
            sample_x = sample_from_model(sess, data)
            test_data.reset()

            img_tile = plotting.img_tile(sample_x[:25], aspect_ratio=1.0, border_color=1.0, stretch=True)
            img = plotting.plot_img(img_tile, title=args.data_set + ' samples')
            plotting.plt.savefig(os.path.join(args.save_dir,'%s_lvae_sample%d.png' % (args.data_set, epoch)))
