import os

import sys
import json
import argparse
import time

import numpy as np
import tensorflow as tf

from pixelcnn import nn
from pixelvae import pixel_vae
from utils import plotting
import utils.mfunc as uf
import utils.mask as um
# self define modules
from config import get_config

cfg = get_config('cifar')


parser = argparse.ArgumentParser()


# vae
parser.add_argument('-is', '--img_size', type=int, default=64, help="size of input image")
parser.add_argument('-zd', '--z_dim', type=int, default=32, help="dimension of the latent variable z")
parser.add_argument('-l', '--lam', type=float, default=1., help="threshold under which the KL divergence will not be punished")
parser.add_argument('-b', '--beta', type=float, default=1., help="strength of the KL divergence penalty")
parser.add_argument('-nffm', '--nr_final_feature_maps', type=int, default=32, help="number of final feature maps in the vae part")

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


# data place holder
xs = [tf.placeholder(tf.float32, shape=(None, args.img_size, args.img_size, 3)) for i in range(args.nr_gpu)]
ms = [tf.placeholder_with_default(np.ones((args.batch_size, args.img_size, args.img_size), dtype=np.float32), shape=(None, args.img_size, args.img_size)) for i in range(args.nr_gpu)]
mxs = [tf.multiply(xs[i], tf.stack([ms for k in range(3)], axis=-1)) for i in range(args.nr_gpu)]
# zs = [tf.placeholder(tf.float32, shape=(None, args.z_dim)) for i in range(args.nr_gpu)]

# create the model
model_opt = {"z_dim":args.z_dim, "img_size":args.img_size, "nr_final_feature_maps":args.nr_final_feature_maps, "dropout_p":args.dropout_p, "nr_resnet":args.nr_resnet, "nr_filters":args.nr_filters, "nr_logistic_mix":args.nr_logistic_mix}
model = tf.make_template('pixel_vae', pixel_vae)


locs = [None for i in range(args.nr_gpu)]
log_vars = [None for i in range(args.nr_gpu)]
x_hats = [None for i in range(args.nr_gpu)]

for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        locs[i], log_vars[i], x_hats[i] = model(mxs[i], **model_opt)


quit()
# gradients
grads = []
loss_gen = []
loss_gen_test = []
new_x_gen = []
for i in range(args.nr_gpu):
    with tf.device('/gpu:%d' % i):
        # train
        out = model(xs[i], ghs[i], shs[i], ema=None, dropout_p=args.dropout_p, **model_opt)
        mask_tf = None
        if args.context_conditioning:
            mask_tf = shs[i][:, :, :, -1]
        loss_gen.append(loss_fun(tf.stop_gradient(xs[i]), out, masks=mask_tf))

        # gradients
        grads.append(tf.gradients(loss_gen[i], all_params, colocate_gradients_with_ops=True))

        # test
        out = model(xs[i], ghs[i], shs[i], ema=ema, dropout_p=0., **model_opt)
        loss_gen_test.append(loss_fun(xs[i], out, masks=mask_tf))

        # sample
        out = model(xs[i], gh_sample[i], sh_sample[i], ema=ema, dropout_p=0, **model_opt)
        if args.energy_distance:
            new_x_gen.append(out[0])
        else:
            epsilon = 0.05 #1e-5
            new_x_gen.append(nn.sample_from_discretized_mix_logistic(out, args.nr_logistic_mix, epsilon=epsilon))

# add losses and gradients together and get training updates
tf_lr = tf.placeholder(tf.float32, shape=[])
with tf.device('/gpu:0'):
    for i in range(1,args.nr_gpu):
        loss_gen[0] += loss_gen[i]
        loss_gen_test[0] += loss_gen_test[i]
        for j in range(len(grads[0])):
            grads[0][j] += grads[i][j]
    # training op
    optimizer = tf.group(nn.adam_updates(all_params, grads[0], lr=tf_lr, mom1=0.95, mom2=0.9995), maintain_averages_op)

# convert loss to bits/dim
bits_per_dim = loss_gen[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)
bits_per_dim_test = loss_gen_test[0]/(args.nr_gpu*np.log(2.)*np.prod(obs_shape)*args.batch_size)

# mask generator
train_mgen = um.RandomRectangleMaskGenerator(obs_shape[0], obs_shape[1], max_ratio=1.0)
#train_mgen = um.CenterMaskGenerator(obs_shape[0], obs_shape[1])
test_mgen = um.RandomRectangleMaskGenerator(obs_shape[0], obs_shape[1], max_ratio=1.0)
sample_mgen = um.CenterMaskGenerator(obs_shape[0], obs_shape[1], 0.875)



def sample_from_model(sess, data=None, **params):
    if type(data) is tuple:
        x,y = data
    else:
        x = data
        y = None
    x = np.cast[np.float32]((x - 127.5) / 127.5) ## preprocessing

    # global conditioning
    if args.global_conditional:
        global_lv = []
        if 'z' in params:
            global_lv.append(params['z'])
        global_lv = np.concatenate(global_lv, axis=-1)

    # spatial conditioning
    if args.spatial_conditional:
        spatial_lv = []

        spatial_lv = np.concatenate(spatial_lv, axis=-1)

    if args.global_conditional:
        global_lv = np.split(global_lv, args.nr_gpu)
        feed_dict.update({ghs[i]: global_lv[i] for i in range(args.nr_gpu)})
    if args.spatial_conditional:
        spatial_lv = np.split(spatial_lv, args.nr_gpu)
        feed_dict.update({shs[i]: spatial_lv[i] for i in range(args.nr_gpu)})

    x = np.split(x, args.nr_gpu)
    x_gen = [np.zeros_like(x[0]) for i in range(args.nr_gpu)]

    for yi in range(obs_shape[0]):
        for xi in range(obs_shape[1]):
            feed_dict.update({xs[i]: x_gen[i] for i in range(args.nr_gpu)})
            new_x_gen_np = sess.run(new_x_gen, feed_dict=feed_dict)
            for i in range(args.nr_gpu):
                x_gen[i][:,yi,xi,:] = new_x_gen_np[i][:,yi,xi,:]
    return np.concatenate(x_gen, axis=0)


# turn numpy inputs into feed_dict for use with tensorflow
def make_feed_dict(data, init=False, **params):
    if type(data) is tuple:
        x,y = data
    else:
        x = data
        y = None
    x = np.cast[np.float32]((x - 127.5) / 127.5) ## preprocessing

    # global conditioning
    if args.global_conditional:
        global_lv = []
        if 'z' in params:
            global_lv.append(params['z'])
        global_lv = np.concatenate(global_lv, axis=-1)

    # spatial conditioning
    if args.spatial_conditional:
        spatial_lv = []

        spatial_lv = np.concatenate(spatial_lv, axis=-1)

    if init:
        feed_dict = {x_init: x}
        if args.global_conditional:
            feed_dict.update({gh_init: global_lv})
        if args.spatial_conditional:
            feed_dict.update({sh_init: spatial_lv})
    else:
        x = np.split(x, args.nr_gpu)
        feed_dict = {xs[i]: x[i] for i in range(args.nr_gpu)}
        if args.global_conditional:
            global_lv = np.split(global_lv, args.nr_gpu)
            feed_dict.update({ghs[i]: global_lv[i] for i in range(args.nr_gpu)})
        if args.spatial_conditional:
            spatial_lv = np.split(spatial_lv, args.nr_gpu)
            feed_dict.update({shs[i]: spatial_lv[i] for i in range(args.nr_gpu)})
    return feed_dict

# init & save
initializer = tf.global_variables_initializer()
saver = tf.train.Saver()

# //////////// perform training //////////////
if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)
test_bpd = []
lr = args.learning_rate

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:

    for epoch in range(args.max_epochs):
        begin = time.time()

        # init
        if epoch == 0:
            train_data.reset()  # rewind the iterator back to 0 to do one full epoch
            if args.load_params:
                ckpt_file = args.save_dir + '/params_' + args.data_set + '.ckpt'
                print('restoring parameters from', ckpt_file)
                saver.restore(sess, ckpt_file)
            else:
                print('initializing the model...')
                sess.run(initializer)
                d = train_data.next(args.init_batch_size)
                feed_dict = make_feed_dict(d, init=True)  # manually retrieve exactly init_batch_size examples
                sess.run(init_pass, feed_dict)
            print('starting training')

        # train for one epoch
        train_losses = []
        for d in train_data:
            feed_dict = make_feed_dict(d)
            # forward/backward/update model on each gpu
            lr *= args.lr_decay
            feed_dict.update({ tf_lr: lr })
            l,_ = sess.run([bits_per_dim, optimizer], feed_dict)
            train_losses.append(l)
        train_loss_gen = np.mean(train_losses)

        # compute likelihood over test data
        test_losses = []
        for d in test_data:
            feed_dict = make_feed_dict(d)
            l = sess.run(bits_per_dim_test, feed_dict)
            test_losses.append(l)
        test_loss_gen = np.mean(test_losses)
        test_bpd.append(test_loss_gen)

        # log progress to console
        print("Iteration %d, time = %ds, train bits_per_dim = %.4f, test bits_per_dim = %.4f" % (epoch, time.time()-begin, train_loss_gen, test_loss_gen))
        sys.stdout.flush()

        if epoch % args.save_interval == 0:

            # generate samples from the model
            d = next(test_data)
            sample_x = []
            for i in range(args.num_samples):
                sample_x.append(sample_from_model(sess, data=d)) ##
            sample_x = np.concatenate(sample_x,axis=0)
            img_tile = plotting.img_tile(sample_x[:100], aspect_ratio=1.0, border_color=1.0, stretch=True)
            img = plotting.plot_img(img_tile, title=args.data_set + ' samples')
            plotting.plt.savefig(os.path.join(args.save_dir,'%s_sample%d.png' % (args.data_set, epoch)))
            plotting.plt.close('all')
            np.savez(os.path.join(args.save_dir,'%s_sample%d.npz' % (args.data_set, epoch)), sample_x)

            # save params
            saver.save(sess, args.save_dir + '/params_' + args.data_set + '.ckpt')
            np.savez(args.save_dir + '/test_bpd_' + args.data_set + '.npz', test_bpd=np.array(test_bpd))
