import tensorflow as tf
from tensorflow import keras
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

dataset_path = keras.utils.get_file("auto-mpg.data", "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")

column_names='MPG Cylinders Displacement Horsepower Weight Acceleration Model_Year Origin'.split()
raw_dataset=pd.read_csv(dataset_path, names=column_names, na_values='?', comment='\t',
                        sep=" ", skipinitialspace=True)

dataset=raw_dataset.copy()
dataset=dataset.dropna()

origin=dataset.pop('Origin')

train_dataset=dataset.sample(frac=0.8, random_state=0)
test_dataset=dataset.drop(train_dataset.index)

train_labels=train_dataset.pop('MPG')
test_labels=test_dataset.pop('MPG')


def norm(x): #normalization
  return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


def build_model():
    