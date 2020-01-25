"""Tensorflow basic for Estimator syntax.

Concept 9: Estimator API for Classification

# How to build a TF neural network for Classifiction (Section 6)

## Progress:
    - Define a list of feature column
        - Categorical
        - Metrics
    - create an estimator model
    - create "a data input function"
        + pandas or numpy
    - call train, evaluate, and predict on the estimator object

Reference:
    https://www.udemy.com/course/complete-guide-to-tensorflow-for-deep-learning-with-python
"""

from sklearn.model_selection import train_test_split

import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

diabetes = pd.read_csv('data/pima-indians-diabetes.csv')
print(diabetes.columns)

# Normalize the data with min-max
#   Alternatively can normalize using scikit-learn min-max-scaler
cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure',
                'Triceps', 'Insulin', 'BMI', 'Pedigree']

diabetes[cols_to_norm] = diabetes[cols_to_norm]\
    .apply(lambda x: (x - x.min()) / (x.max() - x.min()))
print(diabetes.head())

# Preprocessing before using estimator api

# Build a feature_column (metrics)
num_preg = tf.feature_column.numeric_column("Number_pregnant")
plasma_gluc = tf.feature_column.numeric_column("Glucose_concentration")
dias_press = tf.feature_column.numeric_column("Blood_pressure")
tricep = tf.feature_column.numeric_column("Triceps")
insulin = tf.feature_column.numeric_column("Insulin")
bmi = tf.feature_column.numeric_column("BMI")
diabetes_pedigree = tf.feature_column.numeric_column(
    "Pedigree")
age = tf.feature_column.numeric_column("Age")

# Build a categorical column (categorical)
print(diabetes["Group"].unique())
assigned_group = tf.feature_column\
    .categorical_column_with_vocabulary_list(
        "Group", ['A' 'B' 'C' 'D'])
# Alternatively, using hash_bucket
# assigned_group = tf.feature_column\
#     .categorical_column_with_hash_bucket(
#         "Group", hash_bucket_size=10)

diabetes["Age"].hist(bins=20)
plt.show()

# Notice that, the age can be also categorical
age_bucket = tf.feature_column\
    .bucketized_column(
        age,
        boundaries=[20, 30, 40, 50, 60, 70, 80])

feat_cols = [num_preg, plasma_gluc, dias_press, tricep,
             insulin, bmi, diabetes_pedigree, age_bucket, assigned_group]

# Build Train and Test spliter, using sklearn
x_data = diabetes.drop('Class', axis=1)
labels = diabetes["Class"]
X_train, X_test, y_train, y_test = train_test_split(x_data,
                                                    labels,
                                                    test_size=0.3,
                                                    random_state=101
                                                    )

# Build an input functions for training data,
#   evaluation, and prediction (from pandas)
input_func = tf.estimator.inputs.pandas_input_fn(
    x=X_train,
    y=y_train,
    batch_size=10,
    num_epochs=1000,
    shuffle=True
)

eval_input_func = tf.estimator.inputs.pandas_input_fn(
    x=X_test,
    y=y_test,
    batch_size=10,
    num_epochs=1,
    shuffle=False)

pred_input_func = tf.estimator.inputs.pandas_input_fn(
    x=X_test,
    batch_size=10,
    num_epochs=1,
    shuffle=False
)

# Build a model with Linear Classifier
model = tf.estimator.LinearClassifier(feature_columns=feat_cols,
                                      n_classes=2)

model.train(input_fn=input_func, steps=1000)

eval_results = model.evaluate(eval_input_func)
predictions = model.predict(pred_input_func)
my_pred = list(predictions)
# see prediction and eval_results,
#   Accuracy very low -> need a better classifier


# Build a model with DNN classifier
#   3 layers of 10 hidden units
dnn_model = tf.estimator.DNNClassifier(feature_columns=feat_cols,
                                       hidden_units=[10, 10, 10],
                                       n_classes=2)

# Could be error, because of feature_cols of assigned_group
dnn_model.train(input_fn=input_func, steps=1000)

# The error was using voclabolary which not compatiable with neural network.
#   The type which is compatible are embedding_column or int
embedded_group_col = tf.feature_column.embedding_column(
    assigned_group, dimension=4
)

feat_cols = [num_preg, plasma_gluc, dias_press, tricep,
             insulin, bmi, diabetes_pedigree, age_bucket, embedded_group_col]

# Rebuild the model with new feat_cols
dnn_model = tf.estimator.DNNClassifier(feature_columns=feat_cols,
                                       hidden_units=[10, 10, 10],
                                       n_classes=2)
dnn_model.train(input_fn=input_func, steps=1000)

dnn_eval_results = dnn_model.evaluate(eval_input_func)
dnn_predictions = dnn_model.predict(pred_input_func)
dnn_pred = list(predictions)

# If the accuracy is still low
#   -> increasing layers and hidden units
hidden_units = [10, 10, 20, 20, 20, 10]
dnn_model_2 = tf.estimator.DNNClassifier(feature_columns=feat_cols,
                                         hidden_units=hidden_units,
                                         n_classes=2)
dnn_model_2.train(input_fn=input_func, steps=1000)

dnn_eval_results_2 = dnn_model.evaluate(eval_input_func)
dnn_predictions_2 = dnn_model.predict(pred_input_func)
dnn_pred_2 = list(dnn_predictions_2)
