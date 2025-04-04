import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
from split_dataset import create_train_and_validation_datasets
from build_model import build_model
from mappings import CharMapping

# Define constants
IMG_FOLDER = "/home/invincibleocean/Work/captcha-predictor/dataset"
MODEL_SAVE_PATH = "/home/invincibleocean/Work/captcha-predictor/models/captcha-breaker-1.keras"

def train_and_save_model():
    """
    Builds, trains, and saves the OCR model.
    """
    # Create training and validation datasets
    # X_train, X_val, y_train, y_val = create_train_and_validation_datasets(IMG_FOLDER, crop=True)
    # Create trainig and validation datasets with crop False
    X_train, X_val, y_train, y_val = create_train_and_validation_datasets(IMG_FOLDER, crop=True)
    

    # Build the model
    model = build_model()

    # Train the model
    history = model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=30)

    # Ensure model directory exists
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # Save the trained model
    model.save(MODEL_SAVE_PATH)

    print(f"Model saved successfully at {MODEL_SAVE_PATH}")

    # Generate model predictions
    y_pred = model.predict(X_val)  # Get predictions
    y_pred = np.argmax(y_pred, axis=2)  # Convert softmax output to class indices
    

    # Create num_to_char mapping (reverse of char_to_num)
    num_to_char = CharMapping.get_num_to_char()
    num_to_char['-1'] = 'UKN'  # Handling unknown predictions
    print(num_to_char)

    # Get the total number of validation samples
    total_samples = 10  # Dynamically get total validation samples

    # Define grid layout
    ncol = 5  # Fixed number of columns
    nrow = (total_samples // ncol) + (total_samples % ncol > 0)  # Calculate required rows

    # Adjust figure size dynamically based on sample count
    fig = plt.figure(figsize=(20, nrow * 2.5))  

    # Loop through all validation samples
    for i in range(total_samples):
        fig.add_subplot(nrow, ncol, i+1)  # Create subplot

        plt.imshow(X_val[i].transpose((1, 0, 2)), cmap='gray')  # Display image
        pred_chars = list(map(lambda x: num_to_char[str(x)], y_pred[i]))  # Convert predictions to characters
        plt.title('Prediction : ' + str(pred_chars))  # Set title with prediction
        plt.axis('off')  # Hide axes

    plt.show()

if __name__ == "__main__":
    train_and_save_model()
