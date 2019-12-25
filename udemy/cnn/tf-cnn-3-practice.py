"""Concept 3 - CNN for CIFAR-10
(Exercise of Section 8).

CIFAR can be loaded using:
    https://keras.io/examples/cifar10_cnn/ (small)
    https://www.tensorflow.org/datasets/overview

CNN
    - Initialization (Weights and Bias)
    -
"""

import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import cifar10


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
# import tensorflow_datasets as tfds
# cifar = tfds.load(name="cifar10", split="train", data_dir="data/cifar10/")
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
