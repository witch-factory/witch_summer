import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import datasets, layers, models


fashion_mnist=keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels)=fashion_mnist.load_data()


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
train_images, test_images = train_images/255, test_images/255
train_images = np.expand_dims(train_images, axis=-1) #dimension expansion
test_images = np.expand_dims(test_images, axis=-1)


model=tf.keras.Sequential()
#Block 1
model.add(layers.Conv2D(32,(3,3), padding='same', activation='elu', input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(32,(3,3), padding='same', activation='elu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))

#Block 2
model.add(layers.Conv2D(64,(3,3), padding='same', activation='elu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Dropout(0.25))
model.add(layers.Conv2D(64,(3,3), padding='same', activation='elu'))

model.add(layers.Flatten())

model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
print(model.summary())

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history=model.fit(train_images, train_labels, batch_size=64, epochs=30)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print("test_acc:",test_acc)