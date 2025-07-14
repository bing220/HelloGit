#import os
#os.environ["KERAS_BACKEND"] = "jax"  # Or "torch" or "tensorflow"

import numpy as np 
import pandas as pd 
from mnist_csv import mnist_csv_data 

import keras 
from keras import layers

# Load the MNIST dataset
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# Load mnist data from csv
(x_train, y_train), (x_test, y_test) = mnist_csv_data() 

# Normalize the pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# add channel in last dimention
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print("x_train shape: ", x_train.shape, x_train.dtype)
print("y_train shape: ", y_train.shape, y_train.dtype)
print("x_test shape: ", x_test.shape, x_test.dtype)
print("y_test shape: ", y_test.shape, y_test.dtype)
 
num_classes = 10
input_shape = (28, 28, 1)

# Define the model
model = keras.models.Sequential([
  keras.Input(shape=input_shape),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dropout(0.2),
  layers.Dense(10, activation='softmax')
])

model.summary()

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)

# Evaluate the model
model.evaluate(x_test, y_test, verbose=2)