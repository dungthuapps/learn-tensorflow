"""Module to learn linear regression with Tensorflow."""
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.utils import input_fn_utils
import numpy as np

# Logging
tf.logging.set_verbosity(tf.logging.INFO)

# Classifier XOR of x1, x2
attributes = [[0, 0], [0, 1], [1, 0], [1, 1]]
labels = [0, 1, 1, 0]
data = np.array(attributes, np.float32)
target = np.array(labels)

feature_columns = [tf.contrib.layers.real_valued_column("")]
learning_rate = 0.1
epochs = 10
classifier = tf.contrib.learn.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[3],
    activation_fn=tf.nn.sigmoid,
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate),
    model_dir="model/xor_classifier",
    onfig=tf.contrib.learn.RunConfig(save_checkpoints_secs=1)
)
classifier.fit(data, target, steps=10000)


# Prediction
def test_set():
    return np.array(attributes, np.float32)


predictions = classifier.predict(input_fn=test_set)
index = 0
for i in predictions:
    print(data[index], "vs actual", target[index], "prediction:", i)
    index += 1


# Validation metrics:
# presisions = true_positives / (false_positives + true_positives)
# recall = true_positives / (false_negatives + true_positives)
metrics = ["accuracy", "mean_absolute_error",
           "precision", "recall",
           "false_negatives", "false_positives", "true_positives"]

val_metrics = {}
for i in metrics:
    t = eval(
        "tf.contrib.learn.MetricSpec(" +
        f"metric_fn=tf.contrib.metrics.streaming_{i}," +
        f"prediction_key=tf.contrib.learn.PredictionKey.CLASSES)")
    val_metrics[i] = t

success_metrics = classifier.evaluate(data, target, metrics=val_metrics)

# Monitor, can specify config=tf.contrib.learn.RunConfig(save_checkpoints_secs=1)
# Monitor with tensorboard tensorboard --logdir=.... --debug

val_monitor = tf.contrib.learn.monitors.ValidationMonitor(
    data, target, every_n_steps=20)

# Another examples
# Classifier AND of x1, x2
attributes = [[0, 0], [0, 1], [1, 0], [1, 1]]
labels = [0, 0, 0, 1]
data = np.array(attributes, np.float32)
target = np.array(labels)

feature_columns = [tf.contrib.layers.real_valued_column("")]
learning_rate = 0.1
epochs = 10000
classifier = tf.contrib.learn.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[1],
    activation_fn=tf.nn.sigmoid,
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
)
classifier.fit(data, target, steps=epochs)

# Prediction


def test_set():
    return np.array(attributes, np.float32)


predictions = classifier.predict(input_fn=test_set)
index = 0
for i in predictions:
    print(data[index], "vs actual", target[index], "prediction:", i)
    index += 1


# Classifier OR of x1, x2
attributes = [[0, 0], [0, 1], [1, 0], [1, 1]]
labels = [0, 1, 1, 1]
data = np.array(attributes, np.float32)
target = np.array(labels)

feature_columns = [tf.contrib.layers.real_valued_column("")]
learning_rate = 0.1
epochs = 10000
classifier = tf.contrib.learn.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[1],
    activation_fn=tf.nn.sigmoid,
    optimizer=tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
)
classifier.fit(data, target, steps=epochs)

# Prediction


def test_set():
    return np.array(attributes, np.float32)


predictions = classifier.predict(input_fn=test_set)
index = 0
for i in predictions:
    print(data[index], "vs actual", target[index], "prediction:", i)
    index += 1

# Statistics of model
print("params ", classifier.get_variable_names())
print("total epochs ", classifier.get_variable_value("global_step"))
print("weights from input layer", classifier.get_variable_value(
    'dnn/hiddenlayer_0/weights'))

# Saving model completely
feature_columns = [tf.contrib.layers.real_valued_column(
    "", dimension=(2), dtype=tf.float32)]
feature_spec = tf.contrib.layers.create_feature_spec_for_parsing(
    feature_columns)
serving_input_fn = input_fn_utils.build_parsing_serving_input_fn(feature_spec)
_f = classifier.model_dir + "/complete/"
classifier.export_savedmodel(_f, serving_input_fn, as_text=True)
