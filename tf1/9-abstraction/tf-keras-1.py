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

from tensorflow.contrib.keras import (layers,
                                      #   losses,
                                      #   optimizers,
                                      #   metrics,
                                      #   activations,
                                      )


from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split

from tensorflow.contrib.keras import models

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


# Build tf model with keras
# Here using Sequential API ~ sequence of model
dnn_keras_model = models.Sequential()

dnn_keras_model.add(layers.Dense(units=13, input_dim=13, activation='relu'))
dnn_keras_model.add(layers.Dense(units=13, activation="relu"))
dnn_keras_model.add(layers.Dense(units=13, activation="relu"))

dnn_keras_model.add(layers.Dense(units=3, activation='softmax'))

dnn_keras_model.compile(optimizer="adam",
                        loss="sparse_categorical_crossentropy",
                        metrics=["accuracy"]
                        )
dnn_keras_model.fit(scaled_x_train, y_train, epochs=50)
predictions = dnn_keras_model.predict_classes(scaled_x_test)

# Reporting
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))
