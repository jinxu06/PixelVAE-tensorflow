import os
import sys
import json
import argparse
import time
from pixelcnn.nn_for_cond import adam_updates
import numpy as np
import tensorflow as tf
from utils import plotting
from vae.conv_pixel_vae import ConvPixelVAE
from layers import visualize_samples
from masks import RandomRectangleMaskGenerator, RectangleMaskGenerator, CenterMaskGenerator
from utils.utils import get_trainable_variables

parser = argparse.ArgumentParser()



cfg = {
    "img_size": 32,
    "z_dim": 32,
    "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
    "save_dir": "/data/ziz/jxu/models/conv_pixel_vae_celeba32_mmd_conditioning_1",
    #"save_dir": "/data/ziz/jxu/models/temp",
    "encoder_save_dir": "/data/ziz/jxu/models/conv_vae_celeba32_tc_beta5",
    "data_set": "celeba32",
    "batch_size": 32,
    "nr_gpu": 4,
    #"gpus": "4,5,6,7",
    "learning_rate": 0.0001,
    "nr_resnet": 5,
    "nr_filters": 100,
    "nr_logistic_mix": 10,
    "beta": 1e5,
    "lam": 0.0,
    "save_interval": 10,
    "reg": "mmd",
    "use_mode": "train",
    "mask_type": "rec",
}

# cfg = {
#     "img_size": 32,
#     "z_dim": 32,
#     "data_dir": "/data/ziz/not-backed-up/jxu/CelebA",
#     "save_dir": "/data/ziz/jxu/models/conv_pixel_vae_celeba32_2stage",
#     #"save_dir": "/data/ziz/jxu/models/temp",
#     "encoder_save_dir": "/data/ziz/jxu/models/conv_vae_celeba32_tc_beta5",
#     "data_set": "celeba32",
#     "batch_size": 32,
#     "nr_gpu": 4,
#     #"gpus": "4,5,6,7",
#     "learning_rate": 0.0001,
#     "nr_resnet": 5,
#     "nr_filters": 100,
#     "nr_logistic_mix": 10,
#     "beta": 1,
#     "lam": 0.0,
#     "save_interval": 10,
#     "reg": "kld",
#     "use_mode": "train",
#     "mask_type": "none",
# }



parser.add_argument('-is', '--img_size', type=int, default=cfg['img_size'], help="size of input image")
# data I/O
parser.add_argument('-dd', '--data_dir', type=str, default=cfg['data_dir'], help='Location for the dataset')
parser.add_argument('-sd', '--save_dir', type=str, default=cfg['save_dir'], help='Location for parameter checkpoints and samples')
parser.add_argument('-esd', '--encoder_save_dir', type=str, default=cfg['encoder_save_dir'], help='Location for encoder parameter checkpoints and samples')
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
parser.add_argument('-nr', '--nr_resnet', type=float, default=cfg['nr_resnet'], help="")
parser.add_argument('-nf', '--nr_filters', type=float, default=cfg['nr_filters'], help="")
parser.add_argument('-nlm', '--nr_logistic_mix', type=float, default=cfg['nr_logistic_mix'], help="")
parser.add_argument('-s', '--seed', type=int, default=1, help='Random seed to use')
# new features
parser.add_argument('-d', '--debug', dest='debug', action='store_true', help='Under debug mode?')
parser.add_argument('-fe', '--freeze_encoder', dest='freeze_encoder', action='store_true', help='freeze parameters for the encoder?')
parser.add_argument('-um', '--use_mode', type=str, default=cfg['use_mode'], help='')
parser.add_argument('-mt', '--mask_type', type=str, default=cfg['mask_type'], help='')

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

#train_mgen = CenterMaskGenerator(args.img_size, args.img_size, ratio=1.0)
train_mgen = RandomRectangleMaskGenerator(args.img_size, args.img_size, min_ratio=0.125, max_ratio=1.0)
test_mgen = RectangleMaskGenerator(args.img_size, args.img_size, rec=(8, 24, 24, 8))

xs = [tf.placeholder(tf.float32, shape=(args.batch_size, args.img_size, args.img_size, 3)) for i in range(args.nr_gpu)]
x_bars = [tf.placeholder(tf.float32, shape=(args.batch_size, args.img_size, args.img_size, 3)) for i in range(args.nr_gpu)]
is_trainings = [tf.placeholder(tf.bool, shape=()) for i in range(args.nr_gpu)]
dropout_ps = [tf.placeholder(tf.float32, shape=()) for i in range(args.nr_gpu)]
if args.mask_type.lower()=="none":
    masks = [None for i in range(args.nr_gpu)]
else:
    masks = [tf.placeholder(tf.float32, shape=(args.batch_size, args.img_size, args.img_size)) for i in range(args.nr_gpu)]

pvaes = [ConvPixelVAE(counters={}) for i in range(args.nr_gpu)]
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
    "nr_resnet": args.nr_resnet,
    "nr_filters": args.nr_filters,
    "nr_logistic_mix": args.nr_logistic_mix,
    "sample_range": 3.0,
}
model = tf.make_template('model', ConvPixelVAE.build_graph)

for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        model(pvaes[i], xs[i], x_bars[i], is_trainings[i], dropout_ps[i], masks=masks[i], **model_opt)

if args.use_mode == 'train':
    #all_params = tf.trainable_variables()
    #all_params = get_trainable_variables(["encode_context"], "not in")
    all_params = get_trainable_variables(["encode_context", "pixel_cnn"])

    if args.freeze_encoder:
        all_params = [p for p in all_params if "conv_encoder_" not in p.name]

    grads = []
    for i in range(args.nr_gpu):
        with tf.device('/gpu:%d' % i):
            grads.append(tf.gradients(pvaes[i].loss, all_params, colocate_gradients_with_ops=True))
    with tf.device('/gpu:0'):
        for i in range(1, args.nr_gpu):
            for j in range(len(grads[0])):
                grads[0][j] += grads[i][j]

        loss = tf.add_n([v.loss for v in pvaes]) / args.nr_gpu
        loss_ae = tf.add_n([v.loss_ae for v in pvaes]) / args.nr_gpu
        loss_reg = tf.add_n([v.loss_reg for v in pvaes]) / args.nr_gpu
        train_step = adam_updates(all_params, grads[0], lr=args.learning_rate)



def make_feed_dict(data, is_training=True, dropout_p=0.5):
    data = np.cast[np.float32]((data - 127.5) / 127.5)
    ds = np.split(data, args.nr_gpu)
    feed_dict = {is_trainings[i]: is_training for i in range(args.nr_gpu)}
    feed_dict.update({dropout_ps[i]: dropout_p for i in range(args.nr_gpu)})
    feed_dict.update({ xs[i]:ds[i] for i in range(args.nr_gpu) })
    feed_dict.update({ x_bars[i]:ds[i] for i in range(args.nr_gpu) })
    if masks[0] is not None:
        feed_dict.update({masks[i]:train_mgen.gen(args.batch_size) for i in range(args.nr_gpu)})
    return feed_dict

def sample_from_model(sess, data):
    data = np.cast[np.float32]((data - 127.5) / 127.5)
    ds = np.split(data, args.nr_gpu)
    feed_dict = {is_trainings[i]: False for i in range(args.nr_gpu)}
    feed_dict.update({dropout_ps[i]: 0. for i in range(args.nr_gpu)})
    feed_dict.update({ xs[i]:ds[i] for i in range(args.nr_gpu) })
    tm = test_mgen.gen(args.batch_size)
    if masks[0] is not None:
        feed_dict.update({masks[i]:tm for i in range(args.nr_gpu)})

    x_gen = [ds[i].copy() for i in range(args.nr_gpu)]
    x_gen = [x_gen[i]*np.stack([tm for t in range(3)], axis=-1) for i in range(args.nr_gpu)]
    for yi in range(args.img_size):
        for xi in range(args.img_size):
            if tm[0, yi, xi]==0:
                feed_dict.update({x_bars[i]:x_gen[i] for i in range(args.nr_gpu)})
                x_hats = sess.run([pvaes[i].x_hat for i in range(args.nr_gpu)], feed_dict=feed_dict)
                for i in range(args.nr_gpu):
                    x_gen[i][:, yi, xi, :] = x_hats[i][:, yi, xi, :]
    return np.concatenate(x_gen, axis=0)

def generate_samples(sess, data):
    data = np.cast[np.float32]((data - 127.5) / 127.5)
    ds = np.split(data, args.nr_gpu)
    feed_dict = {is_trainings[i]:False for i in range(args.nr_gpu)}
    feed_dict.update({dropout_ps[i]: 0. for i in range(args.nr_gpu)})
    feed_dict.update({xs[i]:ds[i] for i in range(args.nr_gpu)})
    z_mu = np.concatenate(sess.run([pvaes[i].z_mu for i in range(args.nr_gpu)], feed_dict=feed_dict), axis=0)
    z_log_sigma_sq = np.concatenate(sess.run([pvaes[i].z_log_sigma_sq for i in range(args.nr_gpu)], feed_dict=feed_dict), axis=0)
    z_sigma = np.sqrt(np.exp(z_log_sigma_sq))
    z = np.random.normal(loc=z_mu, scale=z_sigma)
    #z[:, 1] = np.linspace(start=-5., stop=5., num=z.shape[0])
    z = np.split(z, args.nr_gpu)
    feed_dict.update({pvaes[i].z:z[i] for i in range(args.nr_gpu)})

    tm = test_mgen.gen(args.batch_size)
    if masks[0] is not None:
        feed_dict.update({masks[i]:tm for i in range(args.nr_gpu)})

    x_gen = [ds[i].copy() for i in range(args.nr_gpu)]
    x_gen = [x_gen[i]*np.stack([tm for t in range(3)], axis=-1) for i in range(args.nr_gpu)]
    #return np.concatenate(x_gen, axis=0)

    for yi in range(args.img_size):
        for xi in range(args.img_size):
            if tm[0, yi, xi]==0:
                print(yi, xi)
                feed_dict.update({x_bars[i]:x_gen[i] for i in range(args.nr_gpu)})
                x_hats = sess.run([pvaes[i].x_hat for i in range(args.nr_gpu)], feed_dict=feed_dict)
                for i in range(args.nr_gpu):
                    x_gen[i][:, yi, xi, :] = x_hats[i][:, yi, xi, :]
    return np.concatenate(x_gen, axis=0)

def latent_traversal(sess, data, use_image_id=0):
    data = data.copy()
    for i in range(data.shape[0]):
        data[i] = data[use_image_id].copy()
    data = np.cast[np.float32]((data - 127.5) / 127.5)
    ds = np.split(data, args.nr_gpu)
    feed_dict = {is_trainings[i]:False for i in range(args.nr_gpu)}
    feed_dict.update({dropout_ps[i]: 0. for i in range(args.nr_gpu)})
    feed_dict.update({xs[i]:ds[i] for i in range(args.nr_gpu)})
    z_mu = np.concatenate(sess.run([pvaes[i].z_mu for i in range(args.nr_gpu)], feed_dict=feed_dict), axis=0)
    z_log_sigma_sq = np.concatenate(sess.run([pvaes[i].z_log_sigma_sq for i in range(args.nr_gpu)], feed_dict=feed_dict), axis=0)
    z_sigma = np.sqrt(np.exp(z_log_sigma_sq))
    z = np.random.normal(loc=z_mu, scale=z_sigma)
    num_features = 32
    num_traversal_step = 10
    for i in range(num_features):
        z[i*num_traversal_step:(i+1)*num_traversal_step, i] = np.linspace(start=-5., stop=5., num=num_traversal_step)
    z = np.split(z, args.nr_gpu)
    feed_dict.update({pvaes[i].z:z[i] for i in range(args.nr_gpu)})

    x_gen = [ds[i].copy() for i in range(args.nr_gpu)]
    for yi in range(args.img_size):
        for xi in range(args.img_size):
            print(yi, xi)
            feed_dict.update({x_bars[i]:x_gen[i] for i in range(args.nr_gpu)})
            x_hats = sess.run([pvaes[i].x_hat for i in range(args.nr_gpu)], feed_dict=feed_dict)
            for i in range(args.nr_gpu):
                x_gen[i][:, yi, xi, :] = x_hats[i][:, yi, xi, :]
    return np.concatenate(x_gen, axis=0)




initializer = tf.global_variables_initializer()
saver = tf.train.Saver()


var_list = get_trainable_variables(["conv_encoder"])
encoder_saver = tf.train.Saver(var_list=var_list)

#var_list = get_trainable_variables(["conv_encoder", "deconv", "pixel_cnn"])
var_list = get_trainable_variables(["conv_encoder", "deconv"])
saver1 = tf.train.Saver(var_list=var_list)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    sess.run(initializer)

    if args.freeze_encoder:
        encoder_ckpt_file = args.encoder_save_dir + '/params_' + args.data_set + '.ckpt'
        print('restoring encoder parameters from', encoder_ckpt_file)
        encoder_saver.restore(sess, encoder_ckpt_file)

    if args.load_params:
        ckpt_file = args.save_dir + '/params_' + args.data_set + '.ckpt'
        print('restoring parameters from', ckpt_file)
        saver.restore(sess, ckpt_file)

    ckpt_file = "/data/ziz/jxu/models/conv_pixel_vae_celeba32_mmd_nocontext" + '/params_' + args.data_set + '.ckpt'
    print('restoring parameters from', ckpt_file)
    saver1.restore(sess, ckpt_file)

    max_num_epoch = 200
    for epoch in range(max_num_epoch):
        tt = time.time()
        loss_arr, loss_ae_arr, loss_reg_arr = [], [], []
        for data in train_data:
            feed_dict = make_feed_dict(data, is_training=True, dropout_p=0.5)
            _, l, la, lr = sess.run([train_step, loss, loss_ae, loss_reg], feed_dict=feed_dict)
            loss_arr.append(l)
            loss_ae_arr.append(la)
            loss_reg_arr.append(lr)
        train_loss, train_loss_ae, train_loss_reg = np.mean(loss_arr), np.mean(loss_ae_arr), np.mean(loss_reg_arr)

        loss_arr, loss_ae_arr, loss_reg_arr = [], [], []
        for data in test_data:
            feed_dict = make_feed_dict(data, is_training=False, dropout_p=0.)
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

            #visualize_samples(sample_x, os.path.join(args.save_dir,'%s_vae_sample%d.png' % (args.data_set, epoch)), layout=(4,4))

            img_tile = plotting.img_tile(sample_x[:100], aspect_ratio=1.0, border_color=1.0, stretch=True)
            img = plotting.plot_img(img_tile, title=args.data_set + ' samples')
            plotting.plt.savefig(os.path.join(args.save_dir,'%s_pixel_vae_sample%d.png' % (args.data_set, epoch)))
