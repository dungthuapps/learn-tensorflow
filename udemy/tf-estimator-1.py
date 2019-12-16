"""Tensorflow basic estimator syntax.

Concept 8: Estimator API for Regression

* Classical
    - tf.estimator.LinearClassifier
    - tf.estimator.LinearRegressor
* Neural Network
    - tf.estimator.DNNClassifier
    - tf.estimator.DNNRegressor
    - tf.estimator.DNNLinearCombinedClassifer
    - tf.estimator.DNNLinearCombinedRegressor

# How to build a TF neural network for Regression (Section 6)

## Progress:
    - Define a list of feature column
    - create an estimator model
    - create "a data input function"
        + pandas or numpy
    - call train, evaluate, and predict on the estimator object

Reference:
    https://www.udemy.com/course/complete-guide-to-tensorflow-for-deep-learning-with-python
"""

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


# Define feature column
# 1 feature, 1 item of the list,
#   here 1 feature column with single value shape
feat_cols = [tf.feature_column.numeric_column('x', shape=[1])]
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

# Using sklearn to split data
x_data = np.linspace(0., 10., 1000000)
noise = np.random.randn(len(x_data))

# Again with y = mx + b -> assume true is y = 0.5 + 5
y_true = (0.5 * x_data) + 5 + noise

x_train, x_eval, y_train, y_eval = train_test_split(x_data, y_true,
                                                    test_size=0.3,
                                                    random_state=101)
print(x_train.shape, x_eval.shape)

# Create data input function for train
#    our data is numpy, hence using numpy input

# Input function
input_func = tf.estimator.inputs.numpy_input_fn(
    {'x': x_train},  # same name of x column above
    y_train,
    batch_size=8,
    num_epochs=None,    # whole data set 1 time
    shuffle=True

)

# Input func for training, evaluate on training data
train_input_func = tf.estimator.inputs.numpy_input_fn(
    {'x': x_train},  # same name of x column above
    y_train,
    batch_size=8,
    num_epochs=1000,
    shuffle=False

)

# Input funciton for evaluating, evaluate on test data
eval_input_func = tf.estimator.inputs.numpy_input_fn(
    {'x': x_eval},  # same name of x column above
    y_eval,
    batch_size=8,
    num_epochs=1000,
    shuffle=False

)

# Start to train
estimator.train(input_fn=input_func, steps=1000)

# Evalue the model on the train data
train_metrics = estimator.evaluate(input_fn=train_input_func, steps=1000)

# Evalue the model on the test data
eval_metrics = estimator.evaluate(input_fn=eval_input_func, steps=1000)

# Check the result. Train good result, evaluate not-good result -> overfitting
print("Training data metrics")
print(train_metrics)
print(eval_metrics)


# Prediction
new_x_data = np.linspace(0, 10, 10)
input_fn_predict = tf.estimator.inputs.numpy_input_fn(
    {'x': new_x_data}, shuffle=False)

_predictions = estimator.predict(input_fn=input_fn_predict)
predictions = [pred["predictions"] for pred in _predictions]
print(predictions)

# Plot data
x_data.sample(n=250).plot(kind="scatter", x="X Data", y="Y")
plt.plot(new_x_data, predictions, 'r*')
plt.show()
