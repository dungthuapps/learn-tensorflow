"""Abstraction layer in Tensorflow

* Keras
* Layers
* Estimator API
* TF Learn
* TF Slim
...
Core:
    tf.contrib

"""


"""Abstraction: tf.layers"""


import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from tensorflow.contrib.keras import models
import pandas as pd
wine_data = load_wine()
print(type(wine_data), wine_data.keys())
print(wine_data["DESCR"])

feat_data = wine_data["data"]
labels = wine_data["target"]
x_train, x_test, y_train, y_test = train_test_split(feat_data, labels)
print(x_train.shape)

# Normalize
scaler = MinMaxScaler()
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.fit_transform(x_test)


# Build tf model with tf.layers
onehot_y_train = pd.get_dummies(y_train).as_matrix()
onehot_y_test = pd.get_dummies(y_test).as_matrix()

num_feat = 13
num_hidden1 = 13
num_hidden2 = 13
num_outputs = 3

learning_rate = 0.01

x = tf.placeholder(tf.float32, shape=[None, num_feat])
y_true = tf.placeholder(tf.float32, shape=[None, num_outputs])
act_fn = tf.nn.relu

hidden1 = fully_connected(x, num_hidden1, activation_fn=act_fn)
hidden2 = fully_connected(hidden1, num_hidden2, activation_fn=act_fn)

output = fully_connected(hidden2, num_outputs)

loss = tf.losses.softmax_cross_entropy(onehot_labels=y_true, logits=output)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
training_steps = 1000
with tf.Session() as sess:
    sess.run(init)
    for i in range(training_steps):
        sess.run(train, feed_dict={x: scaled_x_train, y_true: onehot_y_train})

    logits = output.eval(feed_dict={x: scaled_x_test})
    preds = tf.argmax(logits, axis=1)
    predictions = preds.eval()

# Reporting
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
