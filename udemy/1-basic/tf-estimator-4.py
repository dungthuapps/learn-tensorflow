"""Tensorflow basic syntax.
Practice with Section 6 - Exercise Part - Clasification
Reference:
    https://www.udemy.com/course/complete-guide-to-tensorflow-for-deep-learning-with-python
"""

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report

import pandas as pd
import tensorflow as tf
# import matplotlib.pyplot as plt


"""Practice 2 - Estimator and Classification (Exercise part)

# Task: Classify

# How-to
    1. Data loading and Insights
    2. Preprocessing -  Splitting and Normalization
    3. Building Feature columns
        - categorical
        - metrics
    4. Building Input functions
    5. Building Model
    6. Train, Evaluate and Predict
    7. Reporting classification errors
    8. Saving and Restore Models

"""

# 1a - Loading data and insights
census = pd.read_csv("data/census_data.csv")
census.head()
census.describe()

# 1b - Spliting data
target_var = "income_bracket"
cat_vars = ["workclass", "education", "marital_status", "occupation",
            "relationship", "race", "gender", "native_country"]
continuous_vars = [c for c in census.columns
                   if c not in cat_vars + [target_var]]

x_data = census.drop(target_var, axis=1)
# convert to 1s to 0s (TF do not understand string)
targets = census[target_var].apply(lambda x: int(x == ' <=50K'))
x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                    targets,
                                                    test_size=0.3,
                                                    random_state=101)

# 2 Normalization for training data - MinMaxScaler data
cols_to_norm = continuous_vars

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(x_train[cols_to_norm])

scaled_x_train = x_train.copy()
scaled_x_test = x_test.copy()

_scaled_train = scaler.transform(x_train[cols_to_norm])
scaled_x_train[cols_to_norm] = pd.DataFrame(_scaled_train,
                                            columns=cols_to_norm,
                                            index=x_train.index)

_scaled_test = scaler.transform(x_test[cols_to_norm])
scaled_x_test[cols_to_norm] = pd.DataFrame(_scaled_test,
                                           columns=cols_to_norm,
                                           index=x_test.index
                                           )

# 3 Building feature_column (metrics)
metric_features = [tf.feature_column.numeric_column(i) for i in cols_to_norm]
categorical_features = []
for c in cat_vars:
    # dimension must be tuned
    # https://www.tensorflow.org/tutorials/structured_data/feature_columns#embedding_columns
    # 1st categorize it with tf
    cat_col = tf.feature_column.\
        categorical_column_with_hash_bucket(c,
                                            hash_bucket_size=10)
    embedding_col = tf.feature_column.embedding_column(cat_col, dimension=10)
    categorical_features.append(embedding_col)
feat_cols = metric_features + categorical_features
keys = continuous_vars + cat_vars

# 4 Building input functions
train_input_func = tf.estimator.inputs.pandas_input_fn(
    x=scaled_x_train,
    y=y_train,
    batch_size=10,
    num_epochs=1000,
    shuffle=True,
)

eval_input_func = tf.estimator.inputs.pandas_input_fn(
    x=scaled_x_test,
    y=y_test,
    batch_size=10,
    num_epochs=1,
    shuffle=False,
)

pred_input_func = tf.estimator.inputs.pandas_input_fn(
    x=scaled_x_test,
    y=y_test,
    batch_size=10,
    num_epochs=1,
    shuffle=False,
)

# 5a Building + a model with Linear Regression
lin_model = tf.estimator.LinearClassifier(feat_cols,
                                          n_classes=len(targets.unique()))

# 6a Train, Evaluate and predictions
lin_model.train(train_input_func, steps=20000)
lin_eval_results = lin_model.evaluate(eval_input_func)
lin_predictions = list(lin_model.predict(pred_input_func))

# 5b Building a model with DNN Regression
hidden_units = [10, 20, 20, 20, 10]
dnn_model = tf.estimator.DNNClassifier(feature_columns=feat_cols,
                                       hidden_units=hidden_units,
                                       n_classes=len(targets.unique()))

# 6b Train, Evaluate and predictions
dnn_model.train(train_input_func, steps=20000)
dnn_eval_results = dnn_model.evaluate(eval_input_func)
dnn_predictions = list(dnn_model.predict(pred_input_func))

# 7 Reporting
final_lin_preds = [pred["class_ids"].item() for pred in lin_predictions]
final_dnn_preds = [pred["class_ids"].item() for pred in dnn_predictions]

lin_pred_report = classification_report(y_test, final_lin_preds)
dnn_pred_report = classification_report(y_test, final_dnn_preds)

print(lin_pred_report)
print(dnn_pred_report)

# 8 Saving models

