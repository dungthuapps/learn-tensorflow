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

"""Abstraction: tf.estimator.

https://www.tensorflow.org/tutorials/estimator/premade
"""
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import estimator
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
import numpy as np
wine_data = load_wine()
print(type(wine_data), wine_data.keys())
print(wine_data["DESCR"])

feat_data = wine_data["data"]
labels = wine_data["target"]
x_train, x_test, y_train, y_test = train_test_split(feat_data, labels)

# Normalize
scaler = MinMaxScaler()
scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.fit_transform(x_test)


# Build tf model with estimator
print(x_train.shape)
feat_cols = [tf.feature_column.numeric_column("x", shape=[13])]

optimier = tf.train.GradientDescentOptimizer(learning_rate=0.01)
deep_model = estimator.DNNClassifier(hidden_units=[10, 10, 10],
                                     feature_columns=feat_cols,
                                     n_classes=len(np.unique(labels)),
                                     optimizer=optimier
                                     )


train_input_fn = estimator.inputs.numpy_input_fn(x={"x": scaled_x_train},
                                                 y=y_train,
                                                 shuffle=True,
                                                 batch_size=10,
                                                 num_epochs=5)
eval_input_fn = estimator.inputs.numpy_input_fn(x={"x": scaled_x_test},
                                                # y=y_test,
                                                shuffle=False)

deep_model.train(input_fn=train_input_fn, steps=500)
preds = list(deep_model.predict(input_fn=eval_input_fn))
predictions = [p["class_ids"][0] for p in preds]

print(classification_report(y_test, predictions))
