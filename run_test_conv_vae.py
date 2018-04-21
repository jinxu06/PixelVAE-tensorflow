import os
import sys
import json
import argparse
import time
import numpy as np
import tensorflow as tf
from blocks.helpers import Recorder, visualize_samples, get_nonlinearity, int_shape
from blocks.optimizers import adam_updates
import data.load_data as load_data
from models.conv_vae import ConvVAE


parser = argparse.ArgumentParser()

cfg_default = {
    "img_size": 64,
    "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
    "data_set": "celeba64",
    "nonlinearity":"relu",
    "batch_size": 512,
    "learning_rate": 0.0005,
    "lam": 0.0,
    "save_interval": 10,
}


cfg = cfg_default
cfg.update({
    "z_dim": 32,
    "save_dir": "/data/ziz/jxu/models/vae_celeba64_tc_z32_b8",
    "beta": 8.0,
    "reg": "tc",
    "use_mode": "test",
})


parser.add_argument('-is', '--img_size', type=int, default=cfg['img_size'], help="size of input image")
# data I/O
parser.add_argument('-dd', '--data_dir', type=str, default=cfg['data_dir'], help='Location for the dataset')
parser.add_argument('-sd', '--save_dir', type=str, default=cfg['save_dir'], help='Location for parameter checkpoints and samples')
parser.add_argument('-ds', '--data_set', type=str, default=cfg['data_set'], help='Can be either cifar|imagenet')
parser.add_argument('-r', '--reg', type=str, default=cfg['reg'], help='regularization type')
parser.add_argument('-si', '--save_interval', type=int, default=cfg['save_interval'], help='Every how many epochs to write checkpoint/samples?')
parser.add_argument('-lp', '--load_params', dest='load_params', action='store_true', help='Restore training from previous model checkpoint?')
parser.add_argument('-bs', '--batch_size', type=int, default=cfg['batch_size'], help='Batch size during training per GPU')
parser.add_argument('-ng', '--nr_gpu', type=int, default=0, help='How many GPUs to distribute the training across?')
parser.add_argument('-g', '--gpus', type=str, default="", help='GPU No.s')
parser.add_argument('-n', '--nonlinearity', type=str, default=cfg['nonlinearity'], help='')
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

args.nr_gpu = len(args.gpus.split(","))
num_traversal_step = 13
args.batch_size = num_traversal_step * (args.z_dim+4) // args.nr_gpu
args.use_mode = 'test'
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
print('input args:\n', json.dumps(vars(args), indent=4, separators=(',',':'))) # pretty print args
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

tf.set_random_seed(args.seed)
batch_size = args.batch_size * args.nr_gpu
data_set = load_data.CelebA(data_dir=args.data_dir, batch_size=batch_size, img_size=args.img_size)
if args.debug:
    train_data = data_set.train(shuffle=True, limit=batch_size*2)
    eval_data = data_set.train(shuffle=True, limit=batch_size*1)
    test_data = data_set.test(shuffle=False, limit=batch_size*1)
else:
    train_data = data_set.train(shuffle=True, limit=-1)
    eval_data = data_set.train(shuffle=True, limit=batch_size*10)
    test_data = data_set.test(shuffle=False, limit=-1)


xs = [tf.placeholder(tf.float32, shape=(args.batch_size, args.img_size, args.img_size, 3)) for i in range(args.nr_gpu)]
is_trainings = [tf.placeholder(tf.bool, shape=()) for i in range(args.nr_gpu)]

vaes = [ConvVAE(counters={}) for i in range(args.nr_gpu)]
model_opt = {
    "use_mode": args.use_mode,
    "z_dim": args.z_dim,
    "reg": args.reg,
    "beta": args.beta,
    "lam": args.lam,
    "N": 200000,
    "nonlinearity": get_nonlinearity(args.nonlinearity),
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

        record_dict = {}
        record_dict['total loss'] = tf.add_n([v.loss for v in vaes]) / args.nr_gpu
        record_dict['recon loss'] = tf.add_n([v.loss_ae for v in vaes]) / args.nr_gpu
        record_dict['mi reg'] = tf.add_n([v.mi for v in vaes]) / args.nr_gpu
        record_dict['tc reg'] = tf.add_n([v.tc for v in vaes]) / args.nr_gpu
        record_dict['dwkld reg'] = tf.add_n([v.dwkld for v in vaes]) / args.nr_gpu
        recorder = Recorder(dict=record_dict, config_str=str(json.dumps(vars(args), indent=4, separators=(',',':'))), log_file=args.save_dir+"/log_file")
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

    z = np.split(z, args.nr_gpu)
    feed_dict.update({vaes[i].z:z[i] for i in range(args.nr_gpu)})
    x_hats = sess.run([vaes[i].x_hat for i in range(args.nr_gpu)], feed_dict=feed_dict)
    return np.concatenate(x_hats, axis=0)




def latent_traversal(sess, data, use_image_id=0):
    data = np.cast[np.float32]((data - 127.5) / 127.5)
    ds = np.split(data, args.nr_gpu)
    feed_dict = {is_trainings[i]:False for i in range(args.nr_gpu)}
    feed_dict.update({xs[i]:ds[i] for i in range(args.nr_gpu)})
    z_mu = np.concatenate(sess.run([vaes[i].z_mu for i in range(args.nr_gpu)], feed_dict=feed_dict), axis=0)
    z_log_sigma_sq = np.concatenate(sess.run([vaes[i].z_log_sigma_sq for i in range(args.nr_gpu)], feed_dict=feed_dict), axis=0)
    z_sigma = np.sqrt(np.exp(z_log_sigma_sq))
    z = np.random.normal(loc=z_mu, scale=z_sigma)
    for i in range(z.shape[0]):
        z[i] = z[use_image_id].copy()

    num_features = args.z_dim
    for i in range(num_features):
        z[i*num_traversal_step:(i+1)*num_traversal_step, i] = np.linspace(start=-6., stop=6., num=num_traversal_step)
    z = np.split(z, args.nr_gpu)
    feed_dict.update({vaes[i].z:z[i] for i in range(args.nr_gpu)})
    x_hats = sess.run([vaes[i].x_hat for i in range(args.nr_gpu)], feed_dict=feed_dict)
    return np.concatenate(x_hats, axis=0)

# def latent_traversal(sess, image, range=[-6, 6], num_traversal_step=13):
#     num_instances = num_traversal_step * args.z_dim
#     data = np.stack([image.copy() for i in range(int(np.ceil(num_instances/float(args.nr_gpu))))], axis=0)
#     data = np.cast[np.float32]((data - 127.5) / 127.5)
#     ds = np.split(data, args.nr_gpu)
#     feed_dict = {is_trainings[i]:False for i in range(args.nr_gpu)}
#     feed_dict.update({dropout_ps[i]: 0. for i in range(args.nr_gpu)})
#     feed_dict.update({xs[i]:ds[i] for i in range(args.nr_gpu)})
#     z_mu = np.concatenate(sess.run([pvaes[i].z_mu for i in range(args.nr_gpu)], feed_dict=feed_dict), axis=0)
#     z_log_sigma_sq = np.concatenate(sess.run([pvaes[i].z_log_sigma_sq for i in range(args.nr_gpu)], feed_dict=feed_dict), axis=0)
#     z_sigma = np.sqrt(np.exp(z_log_sigma_sq))
#     z = np.random.normal(loc=z_mu, scale=z_sigma)
#     for i in range(z.shape[0]):
#         z[i] = z[0].copy()
#     for i in range(args.z_dim):
#         z[i*num_traversal_step:(i+1)*num_traversal_step, i] = np.linspace(start=range[0], stop=range[1], num=num_traversal_step)
#     z = np.split(z, args.nr_gpu)
#     feed_dict.update({vaes[i].z:z[i] for i in range(args.nr_gpu)})
#     x_hats = sess.run([vaes[i].x_hat for i in range(args.nr_gpu)], feed_dict=feed_dict)
#     return np.concatenate(x_hats, axis=0)[:num_instances]




initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    ckpt_file = args.save_dir + '/params_' + args.data_set + '.ckpt'
    print('restoring parameters from', ckpt_file)
    saver.restore(sess, ckpt_file)

    data = test_data.next((args.z_dim+4)*num_traversal_step)
    test_data.reset()
    img = []
    for i in [2, 3, 6]:
        sample_x = latent_traversal(sess, data, use_image_id=i)
        view = visualize_samples(sample_x, None, layout=(args.z_dim+4, num_traversal_step))
        img.append(view.copy())
    img = np.concatenate(img, axis=1)
    from PIL import Image
    img = img.astype(np.uint8)
    img = Image.fromarray(img, 'RGB')
    img.save("/data/ziz/jxu/gpu-results/show_z32_b8.png")

    # data = next(test_data)
    # vdata = np.cast[np.float32]((data - 127.5) / 127.5)
    # visualize_samples(vdata, "results/conv_vae_original.png", layout=(10, 10))
    # sample_x = generate_samples(sess, data)
    # test_data.reset()
    #
    # visualize_samples(sample_x, "results/conv_vae_recon.png", layout=(10, 10))
