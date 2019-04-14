# -*- coding: utf-8 -*-
"""
@author: anantSinghCross
"""
# This code was made just to test things out before I ran the code directly on the server
# You may refer to this if it helps you understand things

import numpy as np
import json
import pandas as pd
from keras.datasets import mnist
from keras.models import model_from_json
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt

json_file = open("model.json","r")
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('model.h5')

global graph
graph = tf.get_default_graph()

print("Model Loaded")

(X_train,Y_train),(X_test,Y_test) = mnist.load_data()
pic1 = X_train[0]
pic2 = X_train[1]
pic1 = cv2.resize(pic1,(150,150))
pic2 = cv2.resize(pic2,(150,150))
cv2.imwrite("picsForTesting/pic1.jpg", pic1)
cv2.imwrite("picsForTesting/pic2.jpg", pic2)

pic = cv2.imread("picsForTesting/pic2.jpg",0)
pic = pic/255
pic = cv2.resize(pic,(28,28))
pic = np.array(pic)
pic = pic.reshape(pic.shape[0]*pic.shape[1])

pic = np.array([pic])
with graph.as_default():
    predictInteger = loaded_model.predict_classes(pic)
    print(predictInteger)
