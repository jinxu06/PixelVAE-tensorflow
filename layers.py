import numpy as np
import tensorflow as tf
from tensorflow.contrib.framework.python.ops import arg_scope, add_arg_scope
from PIL import Image
import utils.mfunc as uf

@add_arg_scope
def conv2d_layer(inputs, num_filters, kernel_size, strides=1, padding='SAME', nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False):
    outputs = tf.layers.conv2d(inputs, num_filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    if bn:
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
    if nonlinearity is not None:
        outputs = nonlinearity(outputs)
    print("    + conv2d_layer", int_shape(inputs), int_shape(outputs), nonlinearity, bn)
    return outputs

@add_arg_scope
def deconv2d_layer(inputs, num_filters, kernel_size, strides=1, padding='SAME', nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False):
    outputs = tf.layers.conv2d_transpose(inputs, num_filters, kernel_size=kernel_size, strides=strides, padding=padding, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    if bn:
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
    if nonlinearity is not None:
        outputs = nonlinearity(outputs)
    print("    + deconv2d_layer", int_shape(inputs), int_shape(outputs), nonlinearity, bn)
    return outputs

@add_arg_scope
def dense_layer(inputs, num_outputs, nonlinearity=None, bn=True, kernel_initializer=None, kernel_regularizer=None, is_training=False):
    inputs_shape = int_shape(inputs)
    assert len(inputs_shape)==2, "inputs should be flattened first"
    outputs = tf.layers.dense(inputs, num_outputs, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer)
    if bn:
        outputs = tf.layers.batch_normalization(outputs, training=is_training)
    if nonlinearity is not None:
        outputs = nonlinearity(outputs)
    print("    + dense_layer", int_shape(inputs), int_shape(outputs), nonlinearity, bn)
    return outputs

def down_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0],1,xs[2],xs[3]]), x[:,:xs[1]-1,:,:]],1)

def right_shift(x):
    xs = int_shape(x)
    return tf.concat([tf.zeros([xs[0],xs[1],1,xs[3]]), x[:,:,:xs[2]-1,:]],2)

def up_shift(x):
    xs = int_shape(x)
    return tf.concat([x[:,1:xs[1],:,:], tf.zeros([xs[0],1,xs[2],xs[3]])],1)

def left_shift(x):
    xs = int_shape(x)
    return tf.concat([x[:,:,1:xs[2],:], tf.zeros([xs[0],xs[1],1,xs[3]])],2)

@add_arg_scope
def down_shifted_conv2d(x, num_filters, filter_size=[2,3], strides=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[filter_size[0]-1,0], [int((filter_size[1]-1)/2),int((filter_size[1]-1)/2)],[0,0]])
    return conv2d_layer(x, num_filters, kernel_size=filter_size, strides=strides, padding='VALID', **kwargs)

@add_arg_scope
def down_shifted_deconv2d(x, num_filters, filter_size=[2,3], strides=[1,1], **kwargs):
    x = deconv2d_layer(x, num_filters, kernel_size=filter_size, strides=strides, padding='VALID', **kwargs)
    xs = int_shape(x)
    return x[:,:(xs[1]-filter_size[0]+1),int((filter_size[1]-1)/2):(xs[2]-int((filter_size[1]-1)/2)),:]

@add_arg_scope
def down_right_shifted_conv2d(x, num_filters, filter_size=[2,2], strides=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[filter_size[0]-1, 0], [filter_size[1]-1, 0],[0,0]])
    return conv2d_layer(x, num_filters, kernel_size=filter_size, strides=strides, padding='VALID', **kwargs)

@add_arg_scope
def down_right_shifted_deconv2d(x, num_filters, filter_size=[2,2], strides=[1,1], **kwargs):
    x = deconv2d_layer(x, num_filters, kernel_size=filter_size, strides=strides, padding='VALID', **kwargs)
    xs = int_shape(x)
    return x[:,:(xs[1]-filter_size[0]+1):,:(xs[2]-filter_size[1]+1),:]

@add_arg_scope
def up_shifted_conv2d(x, num_filters, filter_size=[2,3], strides=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[0, filter_size[0]-1], [int((filter_size[1]-1)/2),int((filter_size[1]-1)/2)],[0,0]])
    return conv2d_layer(x, num_filters, kernel_size=filter_size, strides=strides, padding='VALID', **kwargs)

@add_arg_scope
def up_left_shifted_conv2d(x, num_filters, filter_size=[2,2], strides=[1,1], **kwargs):
    x = tf.pad(x, [[0,0],[0, filter_size[0]-1], [0, filter_size[1]-1],[0,0]])
    return conv2d_layer(x, num_filters, kernel_size=filter_size, strides=strides, padding='VALID', **kwargs)


@add_arg_scope
def nin(x, num_units, **kwargs):
    """ a network in network layer (1x1 CONV) """
    s = int_shape(x)
    x = tf.reshape(x, [np.prod(s[:-1]),s[-1]])
    x = dense_layer(x, num_units, **kwargs)
    return tf.reshape(x, s[:-1]+[num_units])

@add_arg_scope
def gated_resnet(x, a=None, gh=None, sh=None, nonlinearity=tf.nn.elu, conv=conv2d_layer, dropout_p=0.0, counters={}, **kwargs):
    name = get_name("gated_resnet", counters)
    print("construct", name, "...")
    xs = int_shape(x)
    num_filters = xs[-1]
    with arg_scope([conv], **kwargs):
        c1 = conv(nonlinearity(x), num_filters)
        if a is not None: # add short-cut connection if auxiliary input 'a' is given
            c1 += nin(nonlinearity(a), num_filters)
        c1 = nonlinearity(c1)
        c1 = tf.nn.dropout(c1, keep_prob=1. - dropout_p)
        c2 = conv(c1, num_filters * 2)
        # add projection of h vector if included: conditional generation
        if sh is not None:
            c2 += nin(sh, 2*num_filters, nonlinearity=nonlinearity)
        if gh is not None: # haven't finished this part
            pass
        a, b = tf.split(c2, 2, 3)
        c3 = a * tf.nn.sigmoid(b)
        return x + c3




def int_shape(x):
    return list(map(int, x.get_shape()))

def _log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keep_dims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x-m2), axis))

def log_sum_exp(x, axis):
    return tf.reduce_logsumexp(x, axis=axis)

def get_name(layer_name, counters):
    ''' utlity for keeping track of layer names '''
    if not layer_name in counters:
        counters[layer_name] = 0
    name = layer_name + '_' + str(counters[layer_name])
    counters[layer_name] += 1
    return name


def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

def compute_mmd(x, y, sigma_sqr=1.0):
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)
    return mmd

def compute_mi(z, z_mu, z_log_sigma_sq):
    lse_sum, sum_lse = compute_lse_sum_and_sum_lse(z, z_mu, z_log_sigma_sq)
    cond_entropy = tf.reduce_mean(compute_entropy(z_mu, z_log_sigma_sq))
    return -lse_sum - cond_entropy

def compute_dwkld(z, z_mu, z_log_sigma_sq):
    lse_sum, sum_lse = compute_lse_sum_and_sum_lse(z, z_mu, z_log_sigma_sq)
    dist_prior = tf.distributions.Normal(loc=0., scale=1.)
    nll_prior =  tf.reduce_mean(-tf.reduce_sum(dist_prior.log_prob(z), axis=-1))
    return sum_lse + nll_prior

def compute_entropy(z_mu, z_log_sigma_sq):
    batch_size, z_dim = int_shape(z_mu)
    entropy = (tf.reduce_mean(z_log_sigma_sq, axis=1) + tf.log(2*np.pi*np.e)) * z_dim / 2.
    return entropy

def compute_tc(z, z_mu, z_log_sigma_sq):
    return compute_lse_sum_and_sum_lse(z, z_mu, z_log_sigma_sq)
    lse_sum, sum_lse = compute_lse_sum_and_sum_lse(z, z_mu, z_log_sigma_sq)
    return lse_sum - sum_lse

def compute_lse_sum_and_sum_lse(z, z_mu, z_log_sigma_sq):
    z_sigma = tf.sqrt(tf.exp(z_log_sigma_sq))
    log_probs = []
    batch_size, z_dim = int_shape(z_mu)
    z_b = tf.stack([z for i in range(batch_size)], axis=0)
    z_mu_b = tf.stack([z_mu for i in range(batch_size)], axis=1)
    z_sigma_b = tf.stack([z_sigma for i in range(batch_size)], axis=1)
    z_norm = (z_b-z_mu_b) / z_sigma_b

    dist = tf.distributions.Normal(loc=0., scale=1.)
    log_probs = dist.log_prob(z_norm)
    return log_probs
    lse_sum = tf.reduce_mean(log_sum_exp(tf.reduce_sum(log_probs, axis=-1), axis=0))
    sum_lse = tf.reduce_mean(tf.reduce_sum(log_sum_exp(log_probs, axis=0), axis=-1))
    return lse_sum, sum_lse



def visualize_samples(images, name="results/test.png", layout=[5,5], vrange=[-1., 1.]):
    images = (images - vrange[0]) / (vrange[1]-vrange[0]) * 255.
    images = np.rint(images).astype(np.uint8)
    view = uf.tile_images(images, size=layout)
    if name is None:
        return view
    view = Image.fromarray(view, 'RGB')
    view.save(name)

def broadcast_masks_tf(masks, num_channels=None, batch_size=None):
    if num_channels is not None:
        masks = tf.stack([masks for i in range(num_channels)], axis=-1)
    if batch_size is not None:
        masks = tf.stack([masks for i in range(batch_size)], axis=0)
    return masks
