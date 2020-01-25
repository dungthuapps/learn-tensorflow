"""Concept 1: Basic Theory of AutoEncoder (Section 10)
# Dimensionality Reduction ~ PCA
## Structures
    - Input 
        -> linear encoder = hidden layer 
            -> transform data

## Procedures


"""

import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.contrib.layers import fully_connected


# 1 - Prepare dataset
# 1.1 with 3 dimensions
data = make_blobs(n_samples=100, n_features=3)
labels = data[1]

# 1.2 Scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[0])

data_x = scaled_data[:, 0]
data_y = scaled_data[:, 1]
data_z = scaled_data[:, 2]

# 1.3 Plot data in 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(data_x, data_y, data_z, c=labels)
plt.show()


# 2 - Building Model
# 2.1 Set-up
num_inputs = 3
num_hidden = 2
num_outputs = num_inputs

learning_rate = 0.01

# 2.2 Placeholders ~ input
X = tf.placeholder(tf.float32, shape=[None, num_inputs])

# 2.2 Use Existing Layers

# We do not want activation here
hidden = fully_connected(X, num_hidden, activation_fn=None)
outputs = fully_connected(hidden, num_outputs, activation_fn=None)

# 2.3 Loss Function
# Square Errors
#   Our purpose is to dimensional reduction.
#   The hidden layer will be responsible to encoding input.
#   An the ouputs of decoding must be same with input
C = tf.square(outputs - X)

# MSE
loss = tf.reduce_mean(C)

# 2.4 Optimizing Loss with Gradient Descent
optimizer = tf.train.AdamOptimizer(learning_rate)
train = optimizer.minimize(loss)


# 3 - Train Model
# 3.1 Init global variables ~ path

init = tf.global_variables_initializer()
num_steps = 1000

with tf.Session() as sess:
    sess.run(init)

    for iteration in range(num_steps):
        sess.run(train, feed_dict={X: scaled_data})

    # Check outputs of layers, here we convert 3d -> 2d
    output_2d = hidden.eval(feed_dict={X: scaled_data})

# 3.2 Plot result in 2d after reducing dimension
plt.scatter(output_2d[:, 0], output_2d[:, 1], c=data[1])
plt.show()

# 3.3 Interpretation
# ! remember, weights are involved
# For example this new point include 65% of feature 1, 35% of feature 2
