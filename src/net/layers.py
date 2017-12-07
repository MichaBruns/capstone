import tensorflow as tf
import numpy as np

IS_TRAIN_PHASE = tf.placeholder(dtype=tf.bool, name='is_train_phase')

def bn (input, decay=0.9, eps=1e-5, name='bn'):
    with tf.variable_scope(name) as scope:
        bn = tf.cond(IS_TRAIN_PHASE,
            lambda: tf.contrib.layers.batch_norm(input,  decay=decay, epsilon=eps, center=True, scale=True,
                              is_training=1,reuse=None,
                              updates_collections=None, scope=scope),
            lambda: tf.contrib.layers.batch_norm(input, decay=decay, epsilon=eps, center=True, scale=True,
                              is_training=0, reuse=True,
                              updates_collections=None, scope=scope))

    return bn

def concat(input, axis=3, name='cat'):
    cat = tf.concat(axis=axis, values=input, name=name)
    return cat

def conv3d(input, filters, kernel_size, strides, padding, name='Conv3D'):
    with tf.variable_scope(name) as scope:
        block = tf.layers.conv3d(input, filters, kernel_size, strides, padding)
        block = bn(block)
        block = tf.nn.relu(block, name + '_Relu')
    return block

def conv2d(input, filters, kernel_size, strides, padding, name='Conv2D'):
    with tf.variable_scope(name) as scope:
        block = tf.layers.conv2d(input, filters, kernel_size, strides, padding)
        block = bn(block)
        block = tf.nn.relu(block, name + '_Relu')
    return block

def deconv2d(input, filter, output_shape, strides, padding, name='DeConv2D'):
    with tf.variable_scope(name) as scope:
        w = tf.get_variable(name=name + '_weight', shape=filter)
        block = tf.nn.conv2d_transpose(input, w, output_shape, strides, padding)
    return block

#http://warmspringwinds.github.io/tensorflow/tf-slim/2016/11/22/upsampling-and-image-segmentation-with-tensorflow-and-tf-slim/
#https://github.com/MarvinTeichmann/tensorflow-fcn/blob/master/fcn16_vgg.py
#http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/18/image-segmentation-with-tensorflow-using-cnns-and-conditional-random-fields/
def upsample2d(input, factor = 2, output_features=None, has_bias=True, trainable=True, name='upsample2d'):

    def make_upsample_filter(size):
        '''
        Make a 2D bilinear kernel suitable for upsampling of the given (h, w) size.
        '''
        factor = (size + 1) // 2
        if size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
        og = np.ogrid[:size, :size]
        return (1 - abs(og[0] - center) / factor) * \
               (1 - abs(og[1] - center) / factor)

    input_shape = input.get_shape().as_list()
    assert len(input_shape)==4
    N = input_shape[0]
    H = input_shape[1]
    W = input_shape[2]
    C = output_features
    K = input_shape[3]

    size = 2 * factor - factor % 2
    filter = make_upsample_filter(size)
    weights = np.zeros(shape=(size,size,C,K), dtype=np.float32)
    for c in range(C):
        weights[:, :, c, c] = filter
    init= tf.constant_initializer(value=weights, dtype=tf.float32)

    #https://github.com/tensorflow/tensorflow/issues/833
    output_shape=tf.stack([tf.shape(input)[0], tf.shape(input)[1]*factor,tf.shape(input)[2]*factor, tf.shape(input)[3]])#[N, H*factor, W*factor, C],
    w = tf.get_variable(name=name+'_weight', shape=[size, size, C, K], initializer=init, trainable=trainable)
    deconv = tf.nn.conv2d_transpose(name=name, value=input, filter=w, output_shape=output_shape, strides=[1, factor, factor, 1], padding='SAME')

    if has_bias:
        b = tf.get_variable(name=name+'_bias', shape=[K], initializer=tf.constant_initializer(0.0))
        deconv = deconv+b

    return deconv