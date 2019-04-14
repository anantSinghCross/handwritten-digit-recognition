# -*- coding: utf-8 -*-
"""
@author: anantSinghCross
"""
import flask
import json
import numpy as np
from sklearn.externals import joblib
from flask import Flask, render_template, request
from keras.models import model_from_json
import cv2
import tensorflow as tf

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
	return flask.render_template('index.html')

@app.route("/predict",methods = ['POST'])
def make_predictions():
    if request.method == 'POST':
        pic = request.files['image']
        
        # we'll save the image and the read it again
        pic.save("pic.jpg")

        # read the image (the argument 0 converts the colored image to a single channel grayscale)
        pic = cv2.imread("pic.jpg",0)
        
        # resize the image since the model was trained on images of size 28*28
        pic = cv2.resize(pic,(28,28))
        pic = np.array(pic)
        pic = pic/255
        pic = pic.reshape(pic.shape[0]*pic.shape[1])
        
        pic = np.array([pic])
        
        # predict the number
        with graph.as_default():
            pred = loaded_model.predict_classes(pic)
            return flask.render_template('predict.html' , response = pred[0])
    
    return render_template('index.html')
    
    
if __name__ == '__main__':
    
    # loading the model
    json_file = open('model.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    
    # loading the weights
    loaded_model.load_weights('model.h5')
    loaded_model.summary()
    global graph
    graph = tf.get_default_graph()
    app.run(host='0.0.0.0', port=8001, debug=True)
    