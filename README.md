# Handwritten Digit Recognition

Making and training a Neural Networks model (using keras) to recognize handwritten digits.

Currently the model gives an accuracy of **98.18%** which can be improved upto **99.8%** using more complex *Convolutional Neural
Networks*(CNN). Right now I've used a really simple neural network since I've got to learn more about *CNN*s. So, I'll be improving 
this project in the near future.

***

### Sample of Images You Are Dealing With

* First image in the dataset

![first.jpg](first.jpg)

* Second image in the dataset

![second.jpg](second.jpg)

***

## Required Libraries

* `pandas`
* `keras`
* `numpy`
* `matplotlib`

***

## What You Should See

When you run the program using `python trainer.py` you should see the following if you already have the above libraries installed.

* Output

![output.jpg](output.jpg)

***

#### Note

You'll still have to feed a 28 by 28 pixel image to this model since it's not a robust model and does very little pre-processing on it's own. The images in the dataset are in the form of a 2D array which is stored in *numpy* arrays.
I hope to improve the model itself as well as the pre-processing in the future.

## Deployment of the model on Flask (14th April 2019)

I have now deployed the model on web using Python's backend framework Flask

### Extra libraries you will need

* `opencv` (imported as `cv2`)
* `tensorflow`
* `flask`

### Steps to get it up And running

1. The *api.py* makes the server and runs as the backend program. This takes in the image file from the form and uses them for prediction. To run the *api.py* head to the command prompt, change the directory and type python *api.py*.

1. Open the web-browser and go to *http://localhost:8001/* to get your index.html running.

### What you should see

For now I have uploaded two pictures from the dataset itself that you can use for testing the deployed model. *pic1.jpg* has the number **5** and the *pic2.jpg* has the number **0**.

* index.html

![index.html](index.jpg)

* predict.html

![predict.html](predict.jpg)

***

#### Note

In the future I'm hoping to add more pre-processing for predicting the pictures that are not present in the dataset.
Will most probably use thresholding on the images based on colors using OpenCV library.
