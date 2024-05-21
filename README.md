# Image Classification Using Convolutional Neural Networks and Flask

## Project Overview

This project is a machine learning-based image classification application built using Python, TensorFlow, and Flask. The app classifies images into one of six categories: airplane, car, cat, dog, fruit, and person. The project involves training a convolutional neural network (CNN) on a dataset of labeled images and then deploying the model in a web application to make predictions on new images uploaded by users.

## Dataset

The dataset used for this project is stored in the `archive/natural_images` directory and contains images categorized into the following folders:
- `airplane`
- `car`
- `cat`
- `dog`
- `fruit`
- `person`

## Requirements

- Python 3.x
- TensorFlow
- Keras
- Flask
- NumPy
- Matplotlib
- Seaborn
- Pillow

## Installation

1. Clone the repository or download the project files:
    ```bash
    git clone https://github.com/yourusername/ImageClassificationApp.git
    cd ImageClassificationApp
    ```

2. Create a virtual environment and activate it:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3. Install the required packages:
    ```bash
    pip install tensorflow keras flask numpy matplotlib seaborn pillow
    ```

## Project Files

- `archive/natural_images/`: Directory containing the dataset with categorized images.
- `image_classification.py`: Script for training the image classification model.
- `app.py`: Flask web application for uploading images and displaying classification results.
- `templates/index.html`: HTML template for the home page.
- `templates/result.html`: HTML template for displaying the classification result.

## Steps

1. **Data Loading and Preprocessing:**
    - Load and preprocess the images from the dataset.
    - Balance the dataset to ensure equal representation of each class.
    - Split the data into training, validation, and test sets.

2. **Model Definition and Training:**
    - Define and compile the CNN model using TensorFlow and Keras.
    - Train the model using data augmentation for better generalization.
    - Save the trained model to a file.

3. **Model Evaluation:**
    - Evaluate the model on the test set and print the test accuracy.
    - Generate a confusion matrix to visualize the classification performance.

4. **Web Application Deployment:**
    - Develop a Flask web application to upload and classify images.
    - Display the classification result along with the uploaded image.
