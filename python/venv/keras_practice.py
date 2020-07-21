import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings

filename="NNdata_020120.xlsx"

excel_data=pd.read_excel(filename, sheet_name="Sheet1", header=0,
                         names="D0 d0 s0 h0 DF dF sF hF T Vs VmM Yield_strength Young ENE".split(),
                         index_col=None, usecols="A:N")

dataset=excel_data.copy()

train_dataset=dataset.sample(frac=0.8, random_state=0)
test_dataset=dataset.drop(train_dataset.index)

train_stats=train_dataset.describe()
train_stats=train_stats.transpose()


def normalize(x):
    return (x-train_stats['mean'])/train_stats['std']


norm_train_data=normalize(train_dataset)
norm_test_data=normalize(test_dataset)

train_labels=norm_train_data.pop('ENE')
test_labels=norm_test_data.pop('ENE')
print(norm_train_data.keys())


model=keras.Sequential([
    keras.layers.Dense(128, activation='relu', input_shape=[len(norm_train_data.keys())]),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae', 'mse'])
#mae = mean absolute error, mse= mean square error
#build_model 함수로도 구현 가능


class progress(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 100 == 0: print("")
        print('#', end='')


EPOCHS=500
early_stop=keras.callbacks.EarlyStopping(monitor='val_loss', patience=50)

history=model.fit(
    norm_train_data, train_labels, epochs=EPOCHS,
    validation_split=0.2, verbose=0, callbacks=[early_stop,progress()]
)

hist=pd.DataFrame(history.history)
hist['epoch']=history.epoch
print(hist.keys())


def plot_history(history):
    hist=pd.DataFrame(history.history)
    hist['epoch']=history.epoch

    plt.figure()
    plt.subplot(2,1,1)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.plot(hist['epoch'], hist['mean_absolute_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'], label='Validation Error')
    plt.legend()

    plt.subplot(2,1,2)
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error')
    plt.plot(hist['epoch'], hist['mean_squared_error'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'], label='Validation Error')
    plt.legend()
    plt.show()


test_predictions=model.predict(norm_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.axis('equal')
plt.axis('square')
plt.show()

error=test_predictions-test_labels
plt.hist(error, bins=25)
plt.xlabel('Prediction Error')
plt.ylabel('Count')
plt.show()