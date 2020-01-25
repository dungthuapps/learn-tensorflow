"""Concept 1: Examples
# Ideas
1. Create generator G
2. Create discriminator D
3. Try to fool D

# Dataset: MNIST


# Procedures


"""
import logging
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from tqdm import tqdm
from tensorflow.examples.tutorials.mnist import input_data


def generator(z, reuse=None, alpha=0.01):
    """Gen image with 784 size."""
    with tf.variable_scope("gen", reuse=reuse,):
        hidden_1 = tf.layers.dense(inputs=z, units=128)
        hidden_1 = tf.maximum(alpha * hidden_1, hidden_1)
        # ! why max here
        hidden_2 = tf.layers.dense(inputs=hidden_1, units=128)
        hidden_2 = tf.maximum(alpha * hidden_2, hidden_2)

        output = tf.layers.dense(inputs=hidden_2,
                                 units=784,
                                 activation=tf.nn.tanh)
        return output


def discriminator(x, reuse=None, alpha=0.01):
    with tf.variable_scope("dis", reuse=reuse,):
        hidden_1 = tf.layers.dense(inputs=x, units=128)
        hidden_1 = tf.maximum(alpha * hidden_1, hidden_1)

        hidden_2 = tf.layers.dense(inputs=hidden_1, units=128)
        hidden_2 = tf.maximum(alpha * hidden_2, hidden_2)

        logits = tf.layers.dense(inputs=hidden_2, units=1)
        output = tf.sigmoid(logits)
        return output, logits


def loss_func(logits_in, labels_in):
    C = tf.nn.sigmoid_cross_entropy_with_logits(
        logits=logits_in,
        labels=labels_in
    )
    L = tf.reduce_mean(C)
    return L


# 0 - Setup
# 0.1 - Disable all Warnings from TF
logging.disable(logging.WARNING)

# 1 - Data Set
mnist = input_data.read_data_sets("data/mnist/", one_hot=True)
mnist.train.num_examples
mnist.test.num_examples

# 2 - Build Model
# 2.1 - Reset if exists
tf.reset_default_graph()

real_images = tf.placeholder(tf.float32, shape=[None, 784])
noise_images = tf.placeholder(tf.float32, shape=[None, 100])

G = generator(noise_images)

D_ouput_real, D_logits_real = discriminator(real_images)

# reuse=True because 'dis' exists in global path
D_ouput_fake, D_logits_fake = discriminator(G, reuse=True)

# D loss when train data
#   because real all labels = 1 -> real
#   to be more generalize, we apply *0.9 factor (smoothing factor)
D_real_loss = loss_func(D_logits_real,
                        tf.ones_like(D_logits_real)*0.9
                        )
D_fake_loss = loss_func(D_logits_fake,
                        tf.zeros_like(D_logits_real)
                        )

D_loss = D_real_loss + D_fake_loss

G_loss = loss_func(D_logits_fake, tf.ones_like(D_logits_fake))

learning_rate = 0.001

tvars = tf.trainable_variables()

d_vars = [var for var in tvars if 'dis' in var.name]
g_vars = [var for var in tvars if 'gen' in var.name]

# Optimize for both networks
# idea:
#   -> train D to better discriminate fake and real
#   -> train G to better generate to fool D
#   -> both try to make better to compete each other
optimizer = tf.train.AdamOptimizer(learning_rate)
D_trainer = optimizer.minimize(D_loss, var_list=d_vars)
G_trainer = optimizer.minimize(G_loss, var_list=g_vars)

# Train both networks
batch_size = 100
epochs = 30  # intensive computation

init = tf.global_variables_initializer()

# Prepare samples saver
samples = []

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(epochs):
        # How many batches to go through all training samples
        num_batches = mnist.train.num_examples // batch_size
        for i in tqdm(range(num_batches)):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            batch_images = batch_x.reshape(batch_size, 784)
            # ! why * 2 - 1
            #   rescale to make sense for the
            #       hyperbolic tagent activation function of D
            batch_images = batch_images * 2 - 1

            # create noise of 100 pixels
            #   (network will learn -> 784 pixels)
            batch_z = np.random.uniform(-1, 1, size=(batch_size, 100))
            feed_dict = {real_images: batch_images,
                         noise_images: batch_z}

            # we do not care output
            _ = sess.run(D_trainer, feed_dict=feed_dict)
            _ = sess.run(G_trainer, feed_dict={noise_images: batch_z})

        print(f"On Epoch {epoch}")

        # Check how images are generated
        # generate a sample of size 100
        sample_z = np.random.uniform(-1, 1, size=(1, 100))

        # Get (or reuse) trained generator network
        g_net = generator(noise_images, reuse=True)
        g_dict = {noise_images: sample_z}
        gen_sample = sess.run(g_net, feed_dict=g_dict)
        samples.append(gen_sample)

# Presents
# each epoch -> generate 1 sample
#   the later the better it is generated
num_images = len(samples)
cols = num_images // 3
rows = num_images // cols
for i in range(num_images):
    plt.axis('off')
    plt.subplot(rows, cols, i + 1)
    plt.imshow(samples[i].reshape(28, 28))

plt.show()

# ! Only run this when saved a long-trained model
model_path = './models/500_epoch_model.ckpt'
saver = tf.train.Saver(var_list=g_vars)
new_samples = []
with tf.Session() as sess:
    saver.restore(sess, model_path)
    for x in range(5):
        sample_z = np.random.uniform(-1, 1, size=(1, 100))
        g_net = generator(noise_images, reuse=True)
        g_dict = {noise_images: sample_z}
        gen_sample = sess.run(g_net, feed_dict=g_dict)

        new_samples.append(gen_sample)

plt.imgshow(new_samples[0].reshape(28, 28))
plt.show()
