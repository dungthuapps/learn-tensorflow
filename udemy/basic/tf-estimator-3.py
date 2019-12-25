"""Tensorflow basic syntax.
Practice with Section 6 - Exercise Part - Regression
Reference:
    https://www.udemy.com/course/complete-guide-to-tensorflow-for-deep-learning-with-python
"""

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
# import matplotlib.pyplot as plt
import tensorflow as tf

"""Practice 1 - Regression and Estimator (Exercise part)
# Task: Aproximate the median house value of each block

# How-to
    1. Data loading and Insights
    2. Preprocessing -  Splitting and Normalization
    3. Building Feature columns
        - categorical vs metrics
    4. Building Input functions
    5. Building Model
    6. Train, Evaluate and Predict
    7. Calculate the RMSE of models

"""

# 1a - Loading data and insights
cali_housing = pd.read_csv("data/cal_housing_clean.csv")
cali_housing.head()
cali_housing.describe()

# 1b - Spliting data
target_var = "medianHouseValue"
x_data = cali_housing.drop(target_var, axis=1)
targets = cali_housing[target_var]
x_train, x_test, y_train, y_test = train_test_split(x_data,
                                                    targets,
                                                    test_size=0.3,
                                                    random_state=101)

# 2 Normalization for training data - MinMaxScaler data
cols_to_norm = ["housingMedianAge", "totalRooms",  "totalBedrooms",
                "population", "households", "medianIncome"]

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(x_train)

scaled_x_train = scaler.transform(x_train)
scaled_x_train = pd.DataFrame(scaled_x_train,
                              columns=x_train.columns,
                              index=x_train.index)

scaled_x_test = scaler.transform(x_test)
scaled_x_test = pd.DataFrame(scaled_x_test,
                             columns=x_test.columns,
                             index=x_test.index
                             )

# 3 Building feature_column (metrics)
feat_cols = [tf.feature_column.numeric_column(i) for i in cols_to_norm]

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
lin_model = tf.estimator.LinearRegressor(feat_cols)

# 6a Train, Evaluate and predictions
lin_model.train(train_input_func, steps=20000)
lin_eval_results = lin_model.evaluate(eval_input_func)
lin_predictions = list(lin_model.predict(pred_input_func))

# 5b Building a model with DNN Regression
hidden_units = [10, 20, 20, 20, 10]
dnn_model = tf.estimator.DNNRegressor(feature_columns=feat_cols,
                                      hidden_units=hidden_units)

# 6b Train, Evaluate and predictions
dnn_model.train(train_input_func, steps=20000)
dnn_eval_results = dnn_model.evaluate(eval_input_func)
dnn_predictions = list(dnn_model.predict(pred_input_func))


# 7 Calculate RMSE
#   (alternatively, using sklearn.metrics)

final_lin_preds = [pred["predictions"] for pred in lin_predictions]
final_dnn_preds = [pred["predictions"] for pred in dnn_predictions]

# ^0.5 = root of mean square
lin_rmse = mean_squared_error(y_test, final_lin_preds) ** 0.5
dnn_rmse = mean_squared_error(y_test, final_dnn_preds) ** 0.5

print(lin_eval_results)
print(dnn_eval_results)

print(lin_rmse)
print(dnn_rmse)
