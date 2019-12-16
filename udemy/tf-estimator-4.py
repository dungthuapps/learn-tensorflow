"""Tensorflow basic syntax.
Practice with Section 6 - Exercise Part - Clasification
Reference:
    https://www.udemy.com/course/complete-guide-to-tensorflow-for-deep-learning-with-python
"""

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs, fetch_california_housing

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

"""Practice 2 - Classification and Estimator (Exercise part)"""

housing = pd.read_csv("data/cal_housing_clean.csv")

# Fist insight of data
