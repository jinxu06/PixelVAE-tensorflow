import tensorflow as tf

# https://ermongroup.github.io/blog/a-tutorial-on-mmd-variational-autoencoders/

def compute_kernel(x, y):
    x_size = tf.shape(x)[0]
    y_size = tf.shape(y)[0]
    dim = tf.shape(x)[1]
    tiled_x = tf.tile(tf.reshape(x, tf.stack([x_size, 1, dim])), tf.stack([1, y_size, 1]))
    tiled_y = tf.tile(tf.reshape(y, tf.stack([1, y_size, dim])), tf.stack([x_size, 1, 1]))
    return tf.exp(-tf.reduce_mean(tf.square(tiled_x - tiled_y), axis=2) / tf.cast(dim, tf.float32))

def compute_mmd(x, y, sigma_sqr=1.0, batch_size=None):
    assert len(tf.shape[x])==2, "x needs to be flattened first"
    assert len(tf.shape[y])==2, "y needs to be flattened first"
    x_kernel = compute_kernel(x, x)
    y_kernel = compute_kernel(y, y)
    xy_kernel = compute_kernel(x, y)
    mmd = tf.reduce_mean(x_kernel) + tf.reduce_mean(y_kernel) - 2 * tf.reduce_mean(xy_kernel)
    if batch_size is not None:
        mmd = tf.stack([[mmd] for i in range(batch_size)], axis=0)
    return mmd
