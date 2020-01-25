"""Concept 3 - CNN for CIFAR-10
(Exercise of Section 8).

CIFAR can be loaded using:
    https://keras.io/examples/cifar10_cnn/ (small)
    https://www.tensorflow.org/datasets/overview


# Procedures
    1. Load data with one-hot encoding
    2. First Insight of data
    3. Build Models
        1. PlaceHolders of Input X and output y_true
            # ! Notice: X ~ [None, number_pixels]
            - None means to tell TF that, we do not how many images yet, will be done in run time
            - same to y_true
        2. Layer 1 - resize x -> [batch_size, h, w, c]
        3. Layer 2: CNN block 1 (Variable of Weights)
            - neurons: tf.nn.conv2d
            - activations: tf.nn.relu
        4. Flatten of CNN -> DNN (reshape)
        5. DNN layer:
            - neurons: y = w * x + b
            - activations: tf.nn.relu
        6. Dropout Layer:
            - create placeholder to get input from feed_dict
            - tf.nn.dropout
        7. Similar we have flow:
            X -> resize X -> CNN-1 -> MaxPool-1 -> CNN-2 ->
                -> MaxPool-2 -> Flatten (reshape) -> DNN -> Droptout -> Softmax
        8. Cost Function
            - tf.nn.softmax_cross_entropy_with_logits
        9. Lost Function
            - tf.reduce_mean
        10. Optimization (Gradient Descent, back ward update)
            - tf.train.AdamOptimizer
            - optimizer.minimize(.)
    4. Run model (Train + Validation for each epoch)
        1. Specify epochs and batch size
        2. Init global variables (graphs)
        3. Create a session
        4. Inside the session, create a feed_dict to place holders
            2. for each epoch in the session,
                1. feed_dict = random_batch_of (x, y_true, and prob_of_drop_out )
                2. train = run/re-run session with feed_dict
                4. validation for each run
                    1. get number of matches
                    2. get accuracy
                        - tf.reduce_mean of matches
                        - run session with eval_feed_dict

"""

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from keras.datasets import cifar10
from sklearn.preprocessing import OneHotEncoder


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


# 1 - Load Data
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

encoder = OneHotEncoder()

encoder.fit(y_train)
y_train = encoder.transform(y_train).toarray()
y_test = encoder.transform(y_test).toarray()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 2 - First Insights
rand_img = x_train[99]
plt.imshow(rand_img)
plt.show()

# 3 - Build Model

# 3.1 - Place Holders

# get the shape (n, h, w, c), ignore n and convert to [None, 32, 32, 3]
x_shape = [None] + list(x_train.shape[1:])
y_shape = [None] + list(y_train.shape[1:])

x = tf.placeholder(tf.float32, shape=x_shape)
y_true = tf.placeholder(tf.float32, shape=y_shape)
# 3.2 - Resize a Tensor ()
#   skip, no reshape here

# 3.3 - CNN Blocks
_channel = x_shape[-1]
_n_filters = 32
conv2d_1 = convolutional_layer(x, shape=[5, 5, _channel, _n_filters])
pooling_1 = max_pool_2by2(conv2d_1)

_channel = _n_filters
_n_filters = 64
conv2d_2 = convolutional_layer(pooling_1, shape=[5, 5, _channel, _n_filters])
pooling_2 = max_pool_2by2(conv2d_2)

# 3.4 - DNN Block (Final)

_shape = np.prod(pooling_2.get_shape()[1:])
conv2d_flat = tf.reshape(pooling_2, [-1, _shape])
full_layer_1 = tf.nn.relu(normal_full_layer(conv2d_flat, 1024))

hold_prob = tf.placeholder(tf.float32)  # apply drop_out
dropout_1 = tf.nn.dropout(full_layer_1, keep_prob=hold_prob)

# 3.5 - Output
_len_one_hot_encorder = y_shape[-1]
y_pred = normal_full_layer(dropout_1, _len_one_hot_encorder)

# 3.6 - Cost and Lost Function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=y_true,
                                                        logits=y_pred)
L = tf.reduce_mean(cross_entropy)

# 3.7 - Optimization and Backward updating (Gradient Descent)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
train = optimizer.minimize(L)

# 4 - Train and Validation
epochs = 1
batch_size = 50
n_samples = x_train.shape[0]


# 4.1 - Init global variables (TF Graphs)
init = tf.global_variables_initializer()

eval_feed_dict = {x: x_test[:batch_size], y_true: y_test[:batch_size], hold_prob: 1.}

# 4.2 - Train
step = 0
with tf.Session() as sess:
    sess.run(init)
    _t = None
    for i in range(epochs):
        for s in range(0, n_samples, batch_size):
            step += 1
            batch_x = x_train[s: s + batch_size]
            batch_y = y_train[s: s + batch_size]
            train_feed_dict = {x: batch_x, y_true: batch_y, hold_prob: 0.5}
            sess.run(train, feed_dict=train_feed_dict)
           

            # 4.3 - Validation in the same session
            matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
            acc = tf.reduce_mean(tf.cast(matches, tf.float32))

            acc_result = sess.run(acc, feed_dict=eval_feed_dict)
            print(f"STEP {step}: acc {acc_result} \n")


