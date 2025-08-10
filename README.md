Facial Expression Detection
This project detects facial expressions from an image or a real-time video feed.

Dataset
The dataset used for this project is the FER2013 dataset from a Kaggle competition. It consists of 48x48 pixel grayscale images of faces. The faces have been automatically registered so that they are more or less centered and occupy about the same amount of space in each image.

The dataset contains 7 different expressions:

0: Angry

1: Disgusted

2: Fearful

3: Happy

4: Neutral

5: Sad

6: Surprised

Model
A Convolutional Neural Network (CNN) was built to train the model. The model was trained for 50 epochs.

Libraries
TensorFlow

Keras

OpenCV

Pandas

NumPy

Matplotlib

How to Use
Run the Jupyter Notebook Face_expression.ipynb.

The notebook will first train the model and save it as model.h5.

You can then use the notebook to either:

Detect expressions from an image.

Detect expressions from a live webcam feed.
