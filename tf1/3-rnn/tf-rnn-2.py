"""Concept 2: Time series using RNN with TF.

Predict t1 given t0
Example with Sin(x)
Tasks:
    - Prediction Model
    - Generate series given a seed series
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class TimeSeriesData():
    def __init__(self, num_points, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax
        self.num_points = num_points
        self.resolution = (xmax - xmin) / num_points
        self.x_data = np.linspace(xmin, xmax, num_points)

        self.y_true = np.sin(self.x_data)

    def ret_true(self, x_series):
        return np.sin(x_series)

    def next_batch(self, batch_size, steps, return_batch_ts=False):
        # Randomly [batch_size, 1]
        rand_start = np.random.rand(batch_size, 1)

        # Convert to "be on" time series between xmin and xmax
        #   and to inside the self.x_data
        #   ? Still do not under stand this
        scale = self.xmax - self.xmin - (steps * self.resolution)
        ts_start = rand_start * scale

        # generate [batch_size +1, 1]
        #   -> because possible for e.g 2 input x_t0, x_t1
        _x_axis = np.arange(0.0, steps + 1)
        batch_ts = ts_start + _x_axis * self.resolution

        y_batch = np.sin(batch_ts)

        # Formating for RNN for, exp in this ex 2 ts
        #   ts0 ~ original ts ignored the last step, e.g, 0-> 30
        #   ts1 (ts shifted 1 step forward), e.g 1-> 31
        shape = (-1, steps, 1)

        y0 = y_batch[:, :-1].reshape(shape)
        y1 = y_batch[:, 1:].reshape(shape)

        if return_batch_ts:
            return y0, y1, batch_ts
        return y0, y1


# Prepare Time Series data
ts_data = TimeSeriesData(num_points=250, xmin=0, xmax=10)
plt.plot(ts_data.x_data, ts_data.y_true)
plt.show()

num_steps = 30
y1, y2, ts = ts_data.next_batch(1, num_steps, True)

# Check dimensions, y1, y2 has num_steps, but ts has num_steps + 1
print(ts.shape, y1.shape, y2.shape)

# insert 1 value to y2
plt.plot(ts_data.x_data, ts_data.y_true, label="sin(x)")
plt.plot(ts.flatten()[1:], y2.flatten(), '.', label="Batch Y t1")
plt.legend()
plt.tight_layout()
plt.show()

# Prepare Training data for t0, t1 as input, each has num_steps
#   also ensure it will between scales of ts_data (min_max)
_total_steps = num_steps + 1
_min = 5
_max = 5 + ts_data.resolution * _total_steps
train_inst = np.linspace(_min, _max, _total_steps)
x_t0 = train_inst[:-1]
x_t1 = train_inst[1:]
y_true_t0 = ts_data.ret_true(x_t0)
y_true_t1 = ts_data.ret_true(x_t1)

plt.title("A Training Instance")
plt.plot(x_t0, y_true_t0, 'bo',
         markersize=15, alpha=0.5, label="Training")
plt.plot(x_t1, y_true_t1, 'ko',
         markersize=7, alpha=0.5, label="Target")
plt.show()

# ! Remember, the t1 is actually the target.
#  Need to train model to match t1

tf.reset_default_graph()
num_steps = 30
num_inputs = 1
num_neurons = 100
num_outputs = 1
learning_rate = 0.0001
num_train_iterations = 2000
batch_size = 1

# Placeholders of x
x = tf.placeholder(tf.float32, [None, num_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_steps, num_outputs])

# RNN Unit Layer (cell layer)
cell = tf.contrib.rnn.BasicRNNCell(num_units=num_neurons,
                                   activation=tf.nn.relu)
# Ensure it will wrap ouput sequence to target outputs
wrapper_cell = tf.contrib.rnn.OutputProjectionWrapper(cell,
                                                      output_size=num_outputs)

# Dynamic Unrolling of inputs (unroll the recursive loop)
outputs, states = tf.nn.dynamic_rnn(wrapper_cell, x, dtype=tf.float32)

# cost function ~ SE
square_error = tf.square(outputs - y)

# Loss function ~ MSE
loss = tf.reduce_mean(square_error)

# Optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

# Train
init = tf.global_variables_initializer()
saver = tf.train.Saver()
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.87)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
with tf.Session() as sess:
    sess.run(init)
    for iteration in range(num_train_iterations):
        x_batch, y_batch, ts = ts_data.next_batch(batch_size, num_steps, True)

        train_feed_dict = {x: x_batch, y: y_batch}
        sess.run(train, feed_dict=train_feed_dict)

        if iteration % 100 == 0:
            # ? why train and eval use the same feed_dict?
            eval_feed_dict = {x: x_batch, y: y_batch}
            mse = loss.eval(feed_dict=eval_feed_dict)
            print(iteration, "\tMSE", mse)
    saver.save(sess, "model/rnn_ts_model_1")

with tf.Session() as sess:
    saver.restore(sess, "model/rnn_ts_model_1")
    x_pred = np.sin(x_t0.reshape(-1, num_steps, num_inputs))
    y_pred = sess.run(outputs, feed_dict={x: x_pred})


plt.title("Testing the model")
plt.plot(x_t0, y_true_t0, 'bo',
         markersize=15, alpha=0.5, label="Training")
plt.plot(x_t1, y_true_t1, 'ko',
         markersize=7, alpha=0.5, label="Target")

# ? why [0, :, 0]
plt.plot(x_t1, y_pred[0, :, 0], 'r.',
         markersize=10, alpha=0.5, label="Predictions")
plt.xlabel("Time")
plt.legend()
plt.tight_layout()
plt.show()


"""Task 2: Generate TS from a model

from a trained model generate a long time series.

"""
with tf.Session() as sess:
    saver.restore(sess, "model/rnn_ts_model_1")

    # first  seq will equal zero, but later will go to correct format
    zero_seq_seed = [0 for i in range(num_steps)]

    for iteration in range(len(ts_data.x_data) - num_steps):
        x_batch = np.array(zero_seq_seed[-num_steps:])
        x_batch = x_batch.reshape(1, num_steps, 1)
        x_batch = np.sin(x_batch)

        y_pred = sess.run(outputs, feed_dict={x: x_batch})

        # append only last point predicted
        zero_seq_seed.append(y_pred[0, -1, 0])

# Initial points
plt.plot(ts_data.x_data, zero_seq_seed, "b-")
# Since initial points
plt.plot(ts_data.x_data[:num_steps], zero_seq_seed[:num_steps], "r")
plt.xlabel("Time")
plt.ylabel("Y")
plt.show()

# Improve initial points = zero by using data from train

with tf.Session() as sess:
    saver.restore(sess, "model/rnn_ts_model_1")

    # Improvement Here
    training_instance = list(ts_data.y_true[:30])

    for iteration in range(len(ts_data.x_data) - num_steps):
        x_batch = np.array(training_instance[-num_steps:])
        x_batch = x_batch.reshape(1, num_steps, 1)
        x_batch = np.sin(x_batch)

        y_pred = sess.run(outputs, feed_dict={x: x_batch})

        # append only last point predicted
        training_instance.append(y_pred[0, -1, 0])

# Initial points
plt.plot(ts_data.x_data, ts_data.y_true, "b-")
# Since initial points
plt.plot(ts_data.x_data[:num_steps], training_instance[:num_steps], "r")
plt.xlabel("Time")
plt.ylabel("Y")
plt.show()
