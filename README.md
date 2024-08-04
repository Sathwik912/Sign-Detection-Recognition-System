# Sign Language Detection Project

## Overview
This project detects American Sign Language (ASL) gestures using computer vision and machine learning. It includes:

1. **Dataset Preparation**: Extracts hand landmarks from images of ASL gestures (A to Z).
2. **Model Training**: Trains a Random Forest classifier on the preprocessed data.
3. **Real-time Inference**: Detects and recognizes ASL gestures using a webcam.
4. **Web Interface**: Displays the predicted ASL gesture on a webpage.

## Dataset
The dataset comprises images representing ASL letters from A to Z.

- **Dataset**: [ASL Alphabet Dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset)

## Prerequisites
Ensure you have the following installed:
- Python 3.7 or higher
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn
- Pickle

You can install the required libraries using `pip install opencv-python mediapipe numpy scikit-learn pickle-mixin`.

## Project Structure
- `create_dataset.py`: Extracts hand landmarks from ASL images.
- `train_classifier.py`: Trains a Random Forest classifier.
- `inference_classifier.py`: Detects and recognizes ASL gestures using a webcam.
- `templates`: Contains files for the web interface.

## Scripts

### create_dataset.py
This script processes the ASL image dataset to extract hand landmarks using MediaPipe and saves the landmarks to a file.

### train_classifier.py
This script trains a Random Forest classifier using the extracted hand landmarks and saves the trained model to a file.

### inference_classifier.py
This script uses a webcam to capture real-time ASL gestures, processes the frames to extract hand landmarks, and uses the trained model to predict the gesture.

### templates
This folder contains the files for the web interface that displays the predicted ASL gesture. The web interface shows the detected gesture in real-time using a terminal-style loader animation.


## Running the Project

### Step 1: Create Dataset
Run `create_dataset.py` to extract hand landmarks from the ASL images.

### Step 2: Train Classifier
Run `train_classifier.py` to train the Random Forest classifier.

### Step 3: Real-time Inference
Run `inference_classifier.py` to start the webcam and recognize ASL gestures in real-time.

### Step 4: Web Interface
Open `home.html` in your browser to view the web interface. Ensure the webcam is enabled, and the predicted ASL gesture will be displayed in the terminal-style loader.

## Acknowledgments
- **Dataset**: [ASL Alphabet Dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset)
- **Hand Landmark Detection**: [MediaPipe](https://mediapipe.dev/)

## Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

