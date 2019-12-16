"""Tensorflow basic syntax.
Concepts about PlaceHolder, Variable, Graph, Session
Reference:
    https://www.udemy.com/course/complete-guide-to-tensorflow-for-deep-learning-with-python
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


print(tf.__version__)

# Tensor object - Hello World
hello = tf.constant("Hello")
world = tf.constant(" World")
print(hello)

# To print out an object, put into Session
with tf.Session() as sess:
    sess.run(hello + world)

# Tensor obj ~ operations
a = tf.constant(10)
b = tf.constant(5)
a + b
a + b
a + b
with tf.Session() as sess:
    sess.run(a + b)

# Tensor matrix ~ similar to numpy
const = tf.constant(10)
fill_mat = tf.fill((4, 4), value=10)
zeros = tf.zeros((4, 4))
ones = tf.ones((4, 4))
randr_normal = tf.random_normal((4, 4), mean=0., stddev=1.)
randr_uni = tf.random_uniform((4, 4), minval=0, maxval=1)

# Execute a tensor operation
my_ops = [const, fill_mat, zeros, ones, randr_normal, randr_uni]
# Alternative way to run operations -> Interactive Session
#   useful for notebook session
#   assume all commands is "with Session() as sess"
"""Concept 1: How to execute a Tensor obj"""
sess = tf.InteractiveSession()
for op in my_ops:
    print(sess.run(op))
    print("\n")

# Alternative ly
for op in my_ops:
    print(op.eval())
    print("\n")

# Attributes of an tensor: get_shape
a = tf.constant([[1, 2], [3, 4]])
a.get_shape()
b = tf.constant([[10], [100]])
result = tf.matmul(a, b)
sess.run(result)
# or
result.eval()

"""Concept 2: Graphs

Graphs are sets of connected nodes (vertices)
    connections = edges
    each node is an operation of inputs -> output
Graphs are "constructed" and "executed"

"""
# n1 + n2 = n3
n1 = tf.constant(1)
n2 = tf.constant(2)
n3 = n1 + n2
with tf.Session() as sess:
    sess.run(n3)
# or alternatively n3.eval()

# All operations are added into global graph()
print(tf.get_default_graph())

# create new graph
g = tf.Graph()
g1 = tf.get_default_graph()

print(g, g1)
with g.as_default():
    # inside with of graph default
    print(g is tf.get_default_graph())

# outside with of graph default
print(g is tf.get_default_graph())

"""Concept 3a: Variables
During "the optimization process", tf will
    (1) tunes the parameters of the model ~
        e.g. variables weights & bias
        and variables needs to be "initialized"
"""
sess = tf.InteractiveSession()
my_tensor = tf.random_uniform((4, 4), 0, 1)
my_var = tf.Variable(initial_value=my_tensor)

# Here the Variable is on tensor my_tensor or even 0,
#   that is actually not initialized.
#       -> cause error of un-innitialized value variable here
sess.run(my_var)

# instead, we will init all variables through an Operation of init
init = tf.global_variables_initializer()
sess.run(init)

# rerun the tensor variable again
sess.run(my_var)

"""Concept 3b: PlaceHolders.

Placeholers are initially "empty" and are used in training phase
    (1) required to declare expected type, e.g. tf.float32
    (2) optional: shape
"""
# First dim of shape is offen referred to number of sample
#   None ~ if we feed in batch, we do not know who many samples before hand,
#       -> hence None here
shape = (None, 4)
shape = (4, 4)
ph = tf.placeholder(tf.float32, shape=shape)

"""Concept 4: Build a Graph.

Example WX + B = Z -> Act(Z)

Feed-Forward:
    Variable(weight) >> tf.matmul()
    Placehold(X) >> tf.matmul()
    tf.matmul() >> tf.add()
    Variable(b) >> tf.add()
    tf.add >> activation function
Optimization (Back-propagation):
    Add the cost function to optimize the parameters
"""
np.random.seed(101)
tf.set_random_seed(101)

rand_x = np.random.uniform(0, 100, size=(5, 5))
rand_y = np.random.uniform(0, 100, size=(5, 1))

# Build a graph
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

add_op = x + y  # alternatively, tf.add(...)
mul_op = x * y

# Run a session and Feed data
with tf.Session() as sess:
    add_result = sess.run(add_op, feed_dict={x: rand_x, y: rand_y})
    print(add_result)

    mul_result = sess.run(mul_op, feed_dict={x: rand_x, y: rand_y})

"""Concept 5: Simple Feed-forward Neural Network."""

n_features = 10
n_dense_neurons = 3

# Build a graph
shape = (None, n_features)  # (number_samples, n_features)
x = tf.placeholder(tf.float32, shape)
W = tf.Variable(tf.random_normal([n_features, n_dense_neurons]))  # random init
b = tf.Variable(tf.ones([n_dense_neurons]))  # initialized with 1s
xW = tf.matmul(x, W)
z = xW + b  # or tf.add(xW, b)
a = tf.sigmoid(z)  # or tf.nn.relu(...)

init = tf.global_variables_initializer()

# Only feed-forward
with tf.Session() as sess:
    # Initialized values of variables
    sess.run(init)
    layer_out = sess.run(a, feed_dict={x: np.random.random([1, n_features])})
    print(layer_out)

"""Concept 6: Simple Regression Example"""
# data ~ data + noise
x_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
y_label = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)
plt.plot(x_data, y_label, '.')
plt.show()

# Assume a regession model y = mx + b of y = mx + b
random_m, random_b = np.random.rand(2)
m = tf.Variable(random_m)
b = tf.Variable(random_b)

error = 0
for x, y in zip(x_data, y_label):
    y_hat = m*x + b

    # Cost function ~ Mean Squared Error
    error += (y - y_hat)**2

# Now we optimize the cost
#   min of Loss function over samples (x_data, y_label))
# This is actuall an operation of tensorflow
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    # initialize variables
    sess.run(init)

    # Feed data to placeholder and optimize it.
    training_steps = 100  # try 1 and 100 ~ epochs
    for i in range(training_steps):
        # Start to train
        sess.run(train)

    # Extract out the value of m, b after training
    m_hat, b_hat = sess.run([m, b])

# Now we test data
x_test = np.linspace(-1, 11, 10)
y_pred_plot = m_hat * x_test + b_hat
plt.plot(x_test, y_pred_plot, 'r')  # plot a trend line
plt.plot(x_data, y_label, 'b.')
plt.show()

"""Concept 7: Batching"""
x_data = np.linspace(0., 10., 1000000)
noise = np.random.randn(len(x_data))

# Again with y = mx + b -> assume true is y = 0.5 + 5
y_true = (0.5 * x_data) + 5 + noise

x_df = pd.DataFrame(data=x_data, columns=["X Data"])
y_df = pd.DataFrame(data=y_true, columns=["Y"])

my_data = pd.concat([x_df, y_df], axis=1)  # concatenate columns
my_data.head()

# Sampling data and first insight of data
my_data.sample(n=250).plot(kind="scatter", x="X Data", y="Y")
plt.show()

# Using batch data
batch_size = 8  # n_samples_per_batch
_m, _b = np.random.randn(2)
m = tf.Variable(_m, dtype=tf.float32)    # any value at first time
b = tf.Variable(_b, dtype=tf.float32)

# n_samples_per_batch with single value
x = tf.placeholder(tf.float32, [batch_size])
y = tf.placeholder(tf.float32, [batch_size])
y_hat = m * x + b

# Cost function ~ mean squared error
error = tf.reduce_sum(tf.square(y - y_hat))

# Select optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

with tf.Session() as sess:
    # Init values
    init = tf.global_variables_initializer()
    sess.run(init)

    # Train model
    batches = 10000
    for i in range(batches):
        # Randomly choosing from data ~ can use scikit-test|train split
        rand_ind = np.random.randint(len(x_data), size=batch_size)
        batch_x_data = x_data[rand_ind]
        batch_y_data = y_true[rand_ind]

        sess.run(train, feed_dict={x: batch_x_data, y: batch_y_data})

    # Extract data
    m_hat, b_hat = sess.run([m, b])

# Now we test data
x_test = np.linspace(-1, 11, 10)
y_pred_plot = m_hat * x_test + b_hat
my_data.sample(250).plot(kind="scatter", x="X Data", y="Y")
plt.plot(x_test, y_pred_plot, 'r')  # plot a trend line
plt.show()
