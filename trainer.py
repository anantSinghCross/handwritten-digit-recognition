# -*- coding: utf-8 -*-
"""
@author: anantSinghCross
"""

import pandas as pd
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.utils import np_utils
import matplotlib.pyplot as plt

seed = 7
np.random.seed(seed)

# just for the spyder console window to displa all tuples and cols
pd.options.display.max_columns = 100
pd.options.display.max_rows = 100

(X_train,Y_train),(X_test,Y_test) = mnist.load_data()
print('Dataset has been loaded')
print('Shapes of training dataset are: X=',X_train.shape,'|| Y=',Y_train.shape)

# two samples from the training dataset plotted as images
plt.imshow(X_train[0])
plt.show()
plt.imshow(X_train[1])
plt.show()

# making the dataset fron 3D to 2D
num_pixels = X_train.shape[1]*X_train.shape[2]
X_train = X_train.reshape(X_train.shape[0],num_pixels).astype('float32')
X_test = X_test.reshape(X_test.shape[0],num_pixels).astype('float32')

# we'll now scale the inputs from 0-1 since here it's easy and a good idea.
# colors always range from 0-255 so we'll divide the inputs by 255
X_train = X_train/255
X_test = X_test/255
print('Y_train: ',Y_train)

# this is the one-hot encoding of categories( digits 0-9, here ). If you want to know more you can google it
# for now just remember that most of the ML algorithms don't work without one-hot encoding of categories.
Y_train = np_utils.to_categorical(Y_train)
Y_test = np_utils.to_categorical(Y_test)
print('Number of categories: ',Y_test.shape[1])
num_categories = Y_test.shape[1]

# now we'll define a function where we create a Sequential model
# we'll be building a fairly simple model for this porblem
def baseline_model():
    model = Sequential()
    model.add(Dense(num_pixels , input_dim = num_pixels , kernel_initializer = 'normal' , activation = 'relu'))
    model.add(Dense(num_categories , kernel_initializer = 'normal' , activation = 'softmax'))
    # compiling the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

model = baseline_model()

# fitting the simple neural network model
model.fit(X_train, Y_train, validation_data = (X_test,Y_test),epochs = 10, batch_size = 200, verbose = 1)
scores = model.evaluate(X_test, Y_test, verbose = 1)
print("Baseline Error: %.2f", (100-scores[1]*100))

# now we'll save the keras model just so that we can use it
# whenever we want without the need to retrain it.
model_json = model.to_json()
with open('model.json','w') as json_file:
    json_file.write(model_json)
model.save_weights('model.h5')

print('Model has been saved successfully')
