
-----

# Real-time Facial Expression Recognition

This project uses a **Convolutional Neural Network (CNN)** to detect and classify human facial expressions in real-time from a webcam feed or a static image. The model is trained to recognize seven different emotions: **angry, disgusted, fearful, happy, neutral, sad, and surprised**.

-----

## 📋 Key Features

  - **Real-time Emotion Detection**: Analyzes a live webcam stream to identify faces and predict their expressions frame-by-frame.
  - **Image-Based Analysis**: Classifies the facial expression from any given image file.
  - **Deep Learning Model**: Built with TensorFlow and Keras, implementing a CNN architecture optimized for facial feature extraction.
  - **End-to-End Pipeline**: A complete demonstration from data loading and preprocessing to model training and deployment.

-----

## 🧠 Model & Dataset

The model's architecture is a **Convolutional Neural Network (CNN)**, which is highly effective for image classification tasks. It learns to identify key facial features (like the shape of the mouth or eyes) to make predictions.

  - **Dataset**: The model was trained on the **FER2013** dataset, a popular benchmark for facial expression analysis. It contains over 35,000 grayscale images of faces, each sized at **48x48 pixels**.
  - **Training**: The network was trained for **50 epochs** to achieve robust performance and accurate classification. The final trained model is saved in `model.h5`.

-----

## 🛠️ Technologies Used

  - **Backend**: Python
  - **Deep Learning**: TensorFlow, Keras
  - **Computer Vision**: OpenCV
  - **Data Processing**: NumPy, Pandas
  - **Visualization**: Matplotlib

-----

## 🚀 How to Run

### Prerequisites

Make sure you have Python installed. Then, install the required libraries:

```bash
pip install tensorflow opencv-python pandas numpy matplotlib
```

### Instructions

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/your-username/facial_expression_detection.git
    cd facial_expression_detection
    ```

2.  **Run the Jupyter Notebook**
    Open and run the `Face_expression.ipynb` notebook in a Jupyter environment.

      * **Training**: The notebook will first execute the training cells to build the CNN and train it on the `fer2013.csv` dataset. The trained weights will be saved as `model.h5`.
      * **Prediction**: After training, you can use the later cells in the notebook to:
          * **Test on an Image**: Provide the path to an image to get a classification.
          * **Launch Real-time Detection**: Run the cell that activates the webcam for live expression analysis. Press 'q' to quit the webcam feed.

-----

## 📂 File Descriptions

  - **`Face_expression.ipynb`**: The main Jupyter Notebook containing all the code for data preprocessing, model training, and prediction.
  - **`fer2013.csv`**: The dataset file containing image pixel data and corresponding emotion labels.
  - **`model.h5`**: The saved, pre-trained Keras model file.
  - **`haarcascade_frontalface_default.xml`**: The Haar Cascade file from OpenCV used for detecting the location of faces in an image or video frame.
  - **`README.md`**: You are here\!
