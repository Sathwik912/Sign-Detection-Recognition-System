# Sign Language Detection Project

## Overview
This project detects American Sign Language (ASL) gestures using computer vision and machine learning. It includes:

1. **Dataset Preparation**: Extracts hand landmarks from images of ASL gestures (A to Z).
2. **Model Training**: Trains a Random Forest classifier on the preprocessed data.
3. **Real-time Inference**: Detects and recognizes ASL gestures using a webcam.

## Dataset
The dataset comprises images representing ASL letters from A to Z.

## Prerequisites
- Python 3.7 or higher
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn
- Pickle

## Scripts
- `create_dataset.py`: Extracts hand landmarks from ASL images.
- `train_classifier.py`: Trains a Random Forest classifier.
- `inference_classifier.py`: Detects and recognizes ASL gestures using a webcam.

## Acknowledgments
- Dataset: [ASL Alphabet Dataset](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset)
- MediaPipe library for hand landmark detection.
