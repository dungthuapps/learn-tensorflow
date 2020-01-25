"""Concept 2: Stacked AutoEncoder (Section 10)

# Reconstruct images

## Structures
    - Input image -> encoder -> decoder -> ouput (=input)       

## Procedures


"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data

# 1 - Prepare dataset
mnist = input_data.read_data_sets("data/mnist/", one_hot=True)
mnist.train.num_examples
mnist.test.num_examples


# 2 - Building Model Input()
# 2.1 Set-up

num_inputs = 784    # 28*28 image
num_hiddens = [392, 196, 392]
num_outputs = num_inputs

learning_rate = 0.01

# 2.2 Placeholders ~ input
tf.reset_default_graph()

X = tf.placeholder(tf.float32, shape=[None, num_inputs])
relu = tf.nn.relu

# 2.2 Build variables
# ! Notice: to help scaling from 784 -> 392 (half)
# actually, still random initialize,
#   but in the way faster training and
#   avoid vanishing gradients
initializer = tf.variance_scaling_initializer()

_W1 = initializer([num_inputs, num_hiddens[0]], dtype=tf.float32)
_W2 = initializer([num_hiddens[0], num_hiddens[1]], dtype=tf.float32)
_W3 = initializer([num_hiddens[1], num_hiddens[2]], dtype=tf.float32)
_W4 = initializer([num_hiddens[2], num_outputs], dtype=tf.float32)

_b1 = tf.zeros(num_hiddens[0])
_b2 = tf.zeros(num_hiddens[1])
_b3 = tf.zeros(num_hiddens[2])
_b4 = tf.zeros(num_outputs)

W1 = tf.Variable(_W1)
W2 = tf.Variable(_W2)
W3 = tf.Variable(_W3)
W4 = tf.Variable(_W4)

b1 = tf.Variable(_b1)
b2 = tf.Variable(_b2)
b3 = tf.Variable(_b3)
b4 = tf.Variable(_b4)

hidden_1 = relu(tf.matmul(X, W1) + b1)
hidden_2 = relu(tf.matmul(hidden_1, W2) + b2)
hidden_3 = relu(tf.matmul(hidden_2, W3) + b3)

outputs = relu(tf.matmul(hidden_3, W4) + b4)

# 2.3 Loss Function
# Square Errors
#   Our purpose is to dimensional reduction.
#   The hidden layer will be responsible to encoding input.
#   An the ouputs of decoding must be same with input
C = tf.square(outputs - X)

# loss = MSE over sample set
loss = tf.reduce_mean(C)

# 2.4 Optimizing Loss with Gradient Descent
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)


# 3 - Train Model
# 3.1 Init global variables ~ path
saver = tf.train.Saver()
init = tf.global_variables_initializer()
num_epochs = 5
batch_size = 64
num_batches = mnist.train.num_examples // batch_size

with tf.Session() as sess:
    sess.run(init)

    for epoch in range(num_epochs):
        for iteration in tqdm(range(num_batches)):
            X_batch, y_batch = mnist.train.next_batch(batch_size)
            feed_dict = {X: X_batch}
            sess.run(train, feed_dict=feed_dict)

        training_loss = loss.eval(feed_dict=feed_dict)
        print(f"EPOCH: {epoch} LOSS: {training_loss}")
    saver.save(sess, "model/example_stacked_autoendcoder.ckpt")

# 4 - Testing
# 4.1 Load model and get result vs output
num_test_images = 10
with tf.Session() as sess:
    saver.restore(sess, "model/example_stacked_autoendcoder.ckpt")
    feed_dict = {X: mnist.test.images[:num_test_images]}
    results = outputs.eval(feed_dict=feed_dict)

# 4.2 Compare original images with their reconstructions of images
f, a = plt.subplots(2, 10, figsize=(20, 4))
for i in range(num_test_images):
    # Original images
    _org_img = np.reshape(mnist.test.images[i], (28, 28))
    a[0][i].imshow(_org_img)

    # Recontructed images using ae layer
    _recons_img = np.reshape(results[i], (28, 28))
    a[1][i].imshow(_recons_img)
plt.show()

# 5 How feature are extracted in hidden layers
# 5.1 Reload model but using hidden layer
num_test_images = 10
with tf.Session() as sess:
    saver.restore(sess, "model/example_stacked_autoendcoder.ckpt")
    feed_dict = {X: mnist.test.images[:num_test_images]}
    results = hidden_2.eval(feed_dict=feed_dict)

# 5.2 Replot original with extracted/encoded features
f, a = plt.subplots(2, 10, figsize=(20, 4))
for i in range(num_test_images):
    # Original images
    _org_img = np.reshape(mnist.test.images[i], (28, 28))
    a[0][i].imshow(_org_img)

    # Recontructed images using encoder
    #   (hidden_layer_2  before decoder)
    # ! Notice: the hidden layer 2 has size of 196 = 14x14
    _recons_img = np.reshape(results[i], (14, 14))
    a[1][i].imshow(_recons_img)
plt.show()
