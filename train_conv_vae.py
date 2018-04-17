import os
import sys
import json
import argparse
import time
from pixelcnn.nn import adam_updates
import numpy as np
import tensorflow as tf
from utils import plotting
from vae.conv_vae import ConvVAE

parser = argparse.ArgumentParser()

# cfg = {
#     "img_size": 64,
#     "z_dim": 32,
#     "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
#     "save_dir": "/data/ziz/jxu/models/conv_vae_celeba64_tc_beta1",
#     "data_set": "celeba64",
#     "batch_size": 512,
#     "nr_gpu": 2,
#     #"gpus": "4,5,6,7",
#     "learning_rate": 0.001,
#     "beta": 1.0,
#     "lam": 0.0,
#     "save_interval": 10,
#     "reg": "tc",
#     "use_mode": "train",
# }

cfg = {
    "img_size": 64,
    "z_dim": 32,
    "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
    "save_dir": "/data/ziz/jxu/models/conv_vae_celeba64_tc_beta5",
    "data_set": "celeba64",
    "batch_size": 512,
    "nr_gpu": 2,
    #"gpus": "4,5,6,7",
    "learning_rate": 0.001,
    "beta": 5.0,
    "lam": 0.0,
    "save_interval": 10,
    "reg": "tc",
    "use_mode": "train",
}

# cfg = {
#     "img_size": 64,
#     "z_dim": 32,
#     "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
#     "save_dir": "/data/ziz/jxu/models/conv_vae_celeba64_kld_beta250",
#     "data_set": "celeba64",
#     "batch_size": 512,
#     "nr_gpu": 2,
#     #"gpus": "4,5,6,7",
#     "learning_rate": 0.001,
#     "beta": 250.0,
#     "lam": 0.0,
#     "save_interval": 10,
#     "reg": "kld",
#     "use_mode": "train",
# }


#
cfg = {
    "img_size": 64,
    "z_dim": 10,
    "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
    "save_dir": "/data/ziz/jxu/models/conv_vae_celeba64_tc_beta15",
    "data_set": "celeba64",
    "batch_size": 512,
    "nr_gpu": 2,
    #"gpus": "4,5,6,7",
    "learning_rate": 0.001,
    "beta": 15.0,
    "lam": 0.0,
    "save_interval": 10,
    "reg": "tc",
    "use_mode": "train",
}

cfg = {
    "img_size": 64,
    "z_dim": 16,
    "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
    "save_dir": "/data/ziz/jxu/models/conv_vae_celeba64_tc_beta15_rep",
    "data_set": "celeba64",
    "batch_size": 512,
    "nr_gpu": 2,
    #"gpus": "4,5,6,7",
    "learning_rate": 0.0002,
    "beta": 15.0,
    "lam": 0.0,
    "save_interval": 10,
    "reg": "tc",
    "use_mode": "train",
}




parser.add_argument('-is', '--img_size', type=int, default=cfg['img_size'], help="size of input image")
# data I/O
parser.add_argument('-dd', '--data_dir', type=str, default=cfg['data_dir'], help='Location for the dataset')
parser.add_argument('-sd', '--save_dir', type=str, default=cfg['save_dir'], help='Location for parameter checkpoints and samples')
parser.add_argument('-ds', '--data_set', type=str, default=cfg['data_set'], help='Can be either cifar|imagenet')
parser.add_argument('-r', '--reg', type=str, default=cfg['reg'], help='regularization type')
parser.add_argument('-si', '--save_interval', type=int, default=cfg['save_interval'], help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-lp', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
parser.add_argument('-bs', '--batch_size', type=int, default=cfg['batch_size'], help='Batch size during training per GPU')
parser.add_argument('-ng', '--nr_gpu', type=int, default=cfg['nr_gpu'], help='How many GPUs to distribute the training across?')
parser.add_argument('-lr', '--learning_rate', type=float, default=cfg['learning_rate'], help='Base learning rate')
parser.add_argument('-b', '--beta', type=float, default=cfg['beta'], help="strength of the KL divergence penalty")
parser.add_argument('-l', '--lam', type=float, default=cfg['lam'], help="")
parser.add_argument('-zd', '--z_dim', type=float, default=cfg['z_dim'], help="")
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
# new features
parser.add_argument('-d', '--debug', dest='debug', action='store_true', help='Under debug mode?')
parser.add_argument('-um', '--use_mode', type=str, default=cfg['use_mode'], help='')

args = parser.parse_args()
if args.use_mode == 'test':
    args.debug = True

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
is_trainings = [tf.placeholder(tf.bool, shape=()) for i in range(args.nr_gpu)]

vaes = [ConvVAE(counters={}) for i in range(args.nr_gpu)]
model_opt = {
    "use_mode": args.use_mode,
    "z_dim": args.z_dim,
    "reg": args.reg,
    "beta": args.beta,
    "lam": args.lam,
    "nonlinearity": tf.nn.elu,
    "bn": True,
    "kernel_initializer": tf.contrib.layers.xavier_initializer(),
    "kernel_regularizer": None,
}


model = tf.make_template('model', ConvVAE.build_graph)

for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        model(vaes[i], xs[i],  is_trainings[i], **model_opt)

if args.use_mode == 'train':
    all_params = tf.trainable_variables()
    grads = []
    for i in range(args.nr_gpu):
        with tf.device('/gpu:%d' % i):
            grads.append(tf.gradients(vaes[i].loss, all_params, colocate_gradients_with_ops=True))
    with tf.device('/gpu:0'):
        for i in range(1, args.nr_gpu):
            for j in range(len(grads[0])):
                grads[0][j] += grads[i][j]

        loss = tf.add_n([v.loss for v in vaes]) / args.nr_gpu
        loss_ae = tf.add_n([v.loss_ae for v in vaes]) / args.nr_gpu
        loss_reg = tf.add_n([v.loss_reg for v in vaes]) / args.nr_gpu
        train_step = adam_updates(all_params, grads[0], lr=args.learning_rate)



def make_feed_dict(data, is_training=True):
    data = np.cast[np.float32]((data - 127.5) / 127.5)
    ds = np.split(data, args.nr_gpu)
    feed_dict = {is_trainings[i]: is_training for i in range(args.nr_gpu)}
    feed_dict.update({ xs[i]:ds[i] for i in range(args.nr_gpu) })
    return feed_dict

def sample_from_model(sess, data):
    data = np.cast[np.float32]((data - 127.5) / 127.5)
    ds = np.split(data, args.nr_gpu)
    feed_dict = {is_trainings[i]: False for i in range(args.nr_gpu)}
    feed_dict.update({ xs[i]:ds[i] for i in range(args.nr_gpu) })
    x_hats = sess.run([vaes[i].x_hat for i in range(args.nr_gpu)], feed_dict=feed_dict)
    return np.concatenate(x_hats, axis=0)


def generate_samples(sess, data):
    data = np.cast[np.float32]((data - 127.5) / 127.5)
    ds = np.split(data, args.nr_gpu)
    feed_dict = {is_trainings[i]:False for i in range(args.nr_gpu)}
    feed_dict.update({xs[i]:ds[i] for i in range(args.nr_gpu)})
    z_mu = np.concatenate(sess.run([vaes[i].z_mu for i in range(args.nr_gpu)], feed_dict=feed_dict), axis=0)
    z_log_sigma_sq = np.concatenate(sess.run([vaes[i].z_log_sigma_sq for i in range(args.nr_gpu)], feed_dict=feed_dict), axis=0)
    z_sigma = np.sqrt(np.exp(z_log_sigma_sq))
    z = np.random.normal(loc=z_mu, scale=z_sigma)
    #z[:, 1] = np.linspace(start=-5., stop=5., num=z.shape[0])
    z = np.split(z, args.nr_gpu)
    feed_dict.update({vaes[i].z:z[i] for i in range(args.nr_gpu)})
    x_hats = sess.run([vaes[i].x_hat for i in range(args.nr_gpu)], feed_dict=feed_dict)
    return np.concatenate(x_hats, axis=0)


def latent_traversal(sess, data, use_image_id=0):
    data = data.copy()
    for i in range(data.shape[0]):
        data[i] = data[use_image_id].copy()
    data = np.cast[np.float32]((data - 127.5) / 127.5)
    ds = np.split(data, args.nr_gpu)
    feed_dict = {is_trainings[i]:False for i in range(args.nr_gpu)}
    feed_dict.update({xs[i]:ds[i] for i in range(args.nr_gpu)})
    z_mu = np.concatenate(sess.run([vaes[i].z_mu for i in range(args.nr_gpu)], feed_dict=feed_dict), axis=0)
    z_log_sigma_sq = np.concatenate(sess.run([vaes[i].z_log_sigma_sq for i in range(args.nr_gpu)], feed_dict=feed_dict), axis=0)
    z_sigma = np.sqrt(np.exp(z_log_sigma_sq))
    z = np.random.normal(loc=z_mu, scale=z_sigma)
    num_features = args.z_dim
    num_traversal_step = 10
    for i in range(num_features):
        z[i*num_traversal_step:(i+1)*num_traversal_step, i] = np.linspace(start=-5., stop=5., num=num_traversal_step)
    z = np.split(z, args.nr_gpu)
    feed_dict.update({vaes[i].z:z[i] for i in range(args.nr_gpu)})
    x_hats = sess.run([vaes[i].x_hat for i in range(args.nr_gpu)], feed_dict=feed_dict)
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
    for epoch in range(max_num_epoch+1):
        tt = time.time()
        loss_arr, loss_ae_arr, loss_reg_arr = [], [], []
        for data in train_data:
            feed_dict = make_feed_dict(data, is_training=True)
            _, l, la, lr = sess.run([train_step, loss, loss_ae, loss_reg], feed_dict=feed_dict)
            loss_arr.append(l)
            loss_ae_arr.append(la)
            loss_reg_arr.append(lr)
        train_loss, train_loss_ae, train_loss_reg = np.mean(loss_arr), np.mean(loss_ae_arr), np.mean(loss_reg_arr)

        loss_arr, loss_ae_arr, loss_reg_arr = [], [], []
        for data in test_data:
            feed_dict = make_feed_dict(data, is_training=False)
            l, la, lr = sess.run([loss, loss_ae, loss_reg], feed_dict=feed_dict)
            loss_arr.append(l)
            loss_ae_arr.append(la)
            loss_reg_arr.append(lr)
        test_loss, test_loss_ae, test_loss_reg = np.mean(loss_arr), np.mean(loss_ae_arr), np.mean(loss_reg_arr)

        print("epoch {0} --------------------- Time {1:.2f}s".format(epoch, time.time()-tt))
        print("train loss:{0:.3f}, train ae loss:{1:.3f}, train reg loss:{2:.3f}".format(train_loss, train_loss_ae, train_loss_reg))
        print("test loss:{0:.3f}, test ae loss:{1:.3f}, test reg loss:{2:.3f}".format(test_loss, test_loss_ae, test_loss_reg))
        sys.stdout.flush()

        if epoch % args.save_interval == 0:
            saver.save(sess, args.save_dir + '/params_' + args.data_set + '.ckpt')

            data = next(test_data)
            sample_x = sample_from_model(sess, data)
            test_data.reset()

            img_tile = plotting.img_tile(sample_x[:100], aspect_ratio=1.0, border_color=1.0, stretch=True)
            img = plotting.plot_img(img_tile, title=args.data_set + ' samples')
            plotting.plt.savefig(os.path.join(args.save_dir,'%s_vae_sample%d.png' % (args.data_set, epoch)))
