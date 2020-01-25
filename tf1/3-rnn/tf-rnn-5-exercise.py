"""Concept 3 - RNN
(Exercise of Section 8).

Data:
    https://keras.io/examples/cifar10_cnn/ (small)
    https://www.tensorflow.org/datasets/overview


# Procedures
    1. Load data with one-hot encoding
    2. First Insight of data
    3. Build Models
        1. PlaceHolders of Input X and output y_true
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
                1. feed_dict = random_batch_of (x, y_true, 
                    and prob_of_drop_out )
                2. train = run/re-run session with feed_dict
                4. validation for each run
                    1. get number of matches
                    2. get accuracy
                        - tf.reduce_mean of matches
                        - run session with eval_feed_dict

"""

import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1 - Load data
data = pd.read_csv("data/monthly-milk-production.csv")
data.month = pd.to_datetime(data.month)

# 2 - First Insights of data
data.plot(x="month", y="milk_production")
plt.show()
data = data.set_index("month")

# 3 - Prepare Train / Test split
# Last 12 month
x_test = data.iloc[-12:]
x_train = data.iloc[0:-12]

# scale data
scaler = MinMaxScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)
print(x_train_scaled.shape)
print(x_test_scaled.shape)


# 4 - Create a batch generator for TS
def next_batch(training_data, batch_size, steps):
    # start index should be lower than len - steps
    start = np.random.randint(low=0,
                              high=len(training_data) - steps,
                              size=(batch_size, 1))

    ts_batch = []
    for s in start:
        end = s + steps + 1
        ts_batch.append(training_data[s.item(): end.item()])

    ts_batch_arr = np.array(ts_batch).reshape(-1, steps + 1)
    return ts_batch_arr[:, :-1], ts_batch_arr[:, 1:]


batch_x, batch_y = next_batch(x_train_scaled, batch_size=10, steps=12)


# 5 - Building RNN model

tf.reset_default_graph()
num_steps = 12
num_inputs = 1
num_neurons = 100
num_outputs = 1
learning_rate = 0.03
num_train_iterations = 2000
batch_size = 1

# 5.1 - Placeholders of x
x = tf.placeholder(tf.float32, [None, num_steps, num_inputs])
y = tf.placeholder(tf.float32, [None, num_steps, num_outputs])

# 5.2 - Build RNN layer
rnn_cell = tf.contrib.rnn.BasicRNNCell(num_units=num_neurons,
                                       activation=tf.nn.relu)
# Ensure it will wrap ouput sequence to target outputs
wrapper_cell = tf.contrib.rnn.OutputProjectionWrapper(rnn_cell,
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

# 6 - Train model
init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    for iteration in range(num_train_iterations):
        x_batch, y_batch = next_batch(x_train_scaled, batch_size, num_steps)
        x_batch = x_batch.reshape(-1, num_steps, num_inputs)
        y_batch = y_batch.reshape(-1, num_steps, num_outputs)
        train_feed_dict = {x: x_batch, y: y_batch}
        sess.run(train, feed_dict=train_feed_dict)

        if iteration % 100 == 0:
            # ? why train and eval use the same feed_dict?
            eval_feed_dict = {x: x_batch, y: y_batch}
            mse = loss.eval(feed_dict=eval_feed_dict)
            print(iteration, "\tMSE", mse)
    saver.save(sess, "model/ex_time_series_model")


with tf.Session() as sess:
    saver.restore(sess, "model/ex_time_series_model")
    x_pred = x_test_scaled.reshape(-1, num_steps, num_inputs)
    y_pred = sess.run(outputs, feed_dict={x: x_pred})

y_pred_inversed_scale = scaler.inverse_transform(y_pred.reshape(1, -1))
x_test["generated"] = y_pred_inversed_scale.flatten()

plt.title("Testing the model")
x_test.plot()
plt.show()