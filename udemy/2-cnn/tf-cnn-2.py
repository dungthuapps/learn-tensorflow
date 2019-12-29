"""Concept 2 - CNN for MNIST.

CNN
    - Initialization (Weights and Bias)
    -
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


def init_weights(shape):
    init_random_dist = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init_random_dist)


def init_bias(shape):
    init_bias_vals = tf.constant(0.1, shape=shape)
    return tf.Variable(init_bias_vals)


def conv2d(x, W):
    # input tensor x --> [batch, H, W, C]
    # Kernels --> [H, W, IN, OUT]
    # stride ~ []
    conv = tf.nn.conv2d(x, W,
                        strides=[1, 1, 1, 1],
                        padding="SAME")
    return conv


def max_pool_2by2(x):
    # input tensor x --> [batch, H, W, C]
    # ksize ~ [1, H, W, 1]
    # Pooling has no Weights
    max_pool = tf.nn.max_pool(x,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding="SAME")
    return max_pool


def convolutional_layer(x, shape):
    W = init_weights(shape)
    b = init_bias([shape[3]])
    conv = conv2d(x, W) + b

    # Relu activation
    a = tf.nn.relu(conv)
    return a


def normal_full_layer(input_layer, size):
    input_size = int(input_layer.get_shape()[1])
    W = init_weights([input_size, size])
    b = init_bias([size])
    y = tf.matmul(input_layer, W) + b
    return y


# 1 - Data and Insights
mnist = input_data.read_data_sets("data/mnist/", one_hot=True)
mnist.train.num_examples
mnist.test.num_examples

# 2 - Build model
x = tf.placeholder(tf.float32, shape=[None, 784])
y_true = tf.placeholder(tf.float32, shape=[None, 10])

#   Layer
x_image = tf.reshape(x, [-1, 28, 28, 1])
# 5x5 padding ~ (5, 5, 1) ~ 1 is channels (or color depth)
# 32 filters

conv2d_1 = convolutional_layer(x_image, shape=[5, 5, 1, 32])
pooling_1 = max_pool_2by2(conv2d_1)

# depth of input and filter mus be the same
conv2d_2 = convolutional_layer(pooling_1, shape=[5, 5, 32, 64])
pooling_2 = max_pool_2by2(conv2d_2)

conv2d_flat = tf.reshape(pooling_2, [-1, 7*7*64])
full_layer_1 = tf.nn.relu(normal_full_layer(conv2d_flat, 1024))

#   Regularization by drop out
hold_prob = tf.placeholder(tf.float32)
dropout_1 = tf.nn.dropout(full_layer_1, rate=hold_prob)

#   Ouput is one-hot-encoding vector
y_pred = normal_full_layer(dropout_1, 10)

#   Cost function and Loss
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=y_true,
    logits=y_pred)
L = tf.reduce_mean(cross_entropy)

#   Optimizer (back propogation)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(L)

init = tf.global_variables_initializer()
steps = 5000
batch_size = 50

with tf.Session() as sess:
    sess.run(init)
    eval_feed_dict = {x: mnist.test.images,
                      y_true: mnist.test.labels,
                      hold_prob: 1.}

    for i in range(steps):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        train_feed_dict = {x: batch_x, y_true: batch_y, hold_prob: 0.5}
        sess.run(train, feed_dict=train_feed_dict)

        matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
        acc = tf.reduce_mean(tf.cast(matches, tf.float32))

        acc_result = sess.run(acc, feed_dict=eval_feed_dict)
        if (i % 100) == 0:
            print(f"STEP {i}: acc {acc_result} \n")
