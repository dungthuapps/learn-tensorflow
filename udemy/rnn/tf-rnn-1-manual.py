"""Concept 1: Manual RNN with TF

- 3 Neuron RNN with TF
- 2 time steps model
    . x -> neurons-(w1) -> y -> neurons-(w2) -> y --> output
    . we need 2 weights (2 time steps

Word sequence (ex) : W
t = 0,      1       2       3       4
[   the,    brown,  fox,    is,     quick]
[   the,    red,    fox,    jumped, high]

~ list of list
W[0] = [the, the]
W[1] = [brown, red]
...
W[4] ..

~ -> num_batches = 5, batch_size = 2 samples, time-steps = 5

"""
import numpy as np
import tensorflow as tf

# Constants
num_inputs = 2  # feed 2 time steps
num_neurons = 3

# Placeholders,
#   x0 for all samples of t0 ~ [the, the]
#   x1 for all sample of t1 ~ [brown, red]
#   when (x0, x1) ~ (the, brown)
x0 = tf.placeholder(tf.float32, [None, num_inputs])
x1 = tf.placeholder(tf.float32, [None, num_inputs])

# Variables x-> W1 -> y -> W2 -> y -> output
W1 = tf.Variable(tf.random_normal(shape=[num_inputs, num_neurons]))
W2 = tf.Variable(tf.random_normal(shape=[num_neurons, num_neurons]))
b = tf.Variable(tf.zeros(shape=(1, num_neurons)))

# Graphs
#   x0 -> W1 -> y0
#   (x1 -> W1) and (y0 -> W2) -> y1
# ? Why x1 -> w1
#   Actually, only y back to neuron, x1 reuse the old neurons (unrolled)
y0 = tf.tanh(tf.matmul(x0, W1) + b)
y1 = tf.tanh(tf.matmul(y0, W2) + tf.matmul(x1, W1) + b)

# Create data
# t0, t1
x0_batch = np.array([[0, 1], [2, 3], [4, 5]])
x1_batch = np.array([[100, 101], [102, 103], [104, 105]])

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    feed_dict = {x0: x0_batch, x1: x1_batch}
    y0_output_vals, y1_output_vals = sess.run([y0, y1], feed_dict=feed_dict)

print(y0_output_vals, y1_output_vals)