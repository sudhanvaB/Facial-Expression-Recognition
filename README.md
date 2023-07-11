# Facial Expression Recognition Mini Project

This repository contains a Facial Expression Recognition mini project developed by two 3rd-year students. The project utilizes deep learning techniques to detect and classify facial expressions in real-time using a webcam.

## Introduction

Facial expression recognition is a critical area in computer vision that aims to analyze human emotions through facial images or videos. This project focuses on training a Convolutional Neural Network (CNN) model to recognize seven different facial expressions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The trained model is then used in real-time to detect faces and predict the corresponding emotion.

## Key Features

- Real-time facial expression recognition using a webcam
- Detection and classification of seven facial expressions
- User-friendly graphical interface
- Highly accurate emotion prediction
- Easy to deploy and use

## Dataset

The model is trained on the FER2013 dataset, which contains facial images labeled with emotion categories. The dataset is preprocessed and divided into training, public testing, and private testing sets. The training set is used to train the model, while the public testing set is used for validation during training. The private testing set is used to evaluate the final performance of the trained model.

## Model Architecture

The model architecture is based on a Convolutional Neural Network (CNN) with multiple convolutional and fully connected layers. The model architecture includes:

- Input layer: Accepts grayscale facial images of size 48x48 pixels.
- Convolutional layers: Extracts relevant features from the input images using convolutional filters.
- Batch Normalization layers: Normalizes the activations of the previous layers to improve training stability.
- Activation layers: Applies the Rectified Linear Unit (ReLU) activation function to introduce non-linearity.
- MaxPooling layers: Performs downsampling to reduce the spatial dimensions of the features.
- Dropout layers: Regularizes the model by randomly dropping neurons during training to prevent overfitting.
- Fully Connected layers: Combines the extracted features and performs classification.
- Softmax activation: Generates probability distributions over the emotion categories for classification.

## Evaluation

The model's performance is evaluated using accuracy as the metric. The accuracy is calculated by comparing the predicted emotions with the ground truth emotions in the private testing set. The trained model achieves an accuracy of 55% on the private testing set.

## Usage

To use the Facial Expression Recognition system:

1. Install the required dependencies mentioned in the `requirements.txt` file.
2. Execute the `camera.py` script to open the webcam and start detecting facial expressions in real-time.
3. Ensure proper lighting conditions and position your face within the camera frame for accurate results.
4. The detected facial expression will be displayed on the screen along with a bounding box around the face.

## File Structure

The repository has the following file structure:

- `camera.py`: Python script to start the webcam and perform real-time facial expression recognition.
- `model.py`: Python script containing the FacialExpressionModel class to load and utilize the trained model.
- `face_model.json`: JSON file containing the trained model's architecture.
- `face_model.h5`: HDF5 file containing the trained model's weights.
- `haarcascade_frontalface_default.xml`: XML file containing the pre-trained face cascade classifier for face detection.
- `README.md`: Readme file explaining the project, its features, and usage instructions.

## Acknowledgments

We would like to acknowledge the following resources that helped in the development of this Facial Expression Recognition mini project:

- The FER2013 dataset for training and testing the model.
- The OpenCV library for face detection and image processing.
- The Keras library for building and training the CNN model.
- The Python programming language for implementing the project.

## Conclusion

Facial Expression Recognition is an essential technology with numerous applications in various fields such as human-computer interaction, market research, and mental health. This mini project demonstrates the use of deep learning techniques to accurately detect and classify facial expressions in real-time. The project can serve as a foundation for more advanced facial expression recognition systems or be integrated into other applications to enhance user experiences.
