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
from mappings import CharMapping

# Get the character to number mapping
char_to_num = CharMapping.get_mapping()

##############################################################################################################################
# This function encodes a single sample. 
# Inputs :
# - img_path : the string representing the image path e.g. '/kaggle/input/captcha-version-2-images/samples/samples/6n6gg.jpg'
# - label : the string representing the label e.g. '6n6gg'
# - crop : boolean, if True the image is cropped around the characters and resized to the original size.
# Outputs :
# - a multi-dimensional array reprensenting the image. Its shape is (50, 200, 1)
# - an array of integers representing the label after encoding the characters to integer. E.g [6,16,6,14,14] for '6n6gg' 
##############################################################################################################################
def encode_single_sample(img_path, label, crop):
    # Read image file and returns a tensor with dtype=string
    img = tf.io.read_file(img_path)
    # Decode and convert to grayscale (this conversion does not cause any information lost and reduces the size of the tensor)
    # This decode function returns a tensor with dtype=uint8
    img = tf.io.decode_png(img, channels=1) 
    # Scales and returns a tensor with dtype=float32
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Crop and resize to the original size : 
    # top-left corner = offset_height, offset_width in image = 0, 25 
    # lower-right corner is at offset_height + target_height, offset_width + target_width = 50, 150
    if(crop==True):
        img = tf.image.crop_to_bounding_box(img, offset_height=0, offset_width=10, target_height=50, target_width=180)
        img = tf.image.resize(img,size=[50,200],method='bilinear', preserve_aspect_ratio=False,antialias=False, name=None)
    # Transpose the image because we want the time dimension to correspond to the width of the image.
    img = tf.transpose(img, perm=[1, 0, 2])
    # Converts the string label into an array with 5 integers. E.g. '6n6gg' is converted into [6,16,6,14,14]
    label = list(map(lambda x:char_to_num[x], label))
    return img.numpy(), label

def create_train_and_validation_datasets(img_folder,crop=False, test_size=0.1):
    """
    Creates training and validation datasets from image files in a folder.

    Args:
        img_folder (str): Path to the folder containing image files.
        encode_single_sample (function): Function to encode a single image and its label.
        crop (bool, optional): Whether to crop images. Defaults to False.
        test_size (float, optional): Percentage of data to be used for validation. Defaults to 0.1 (10%).

    Returns:
        tuple: (X_train, X_val, y_train, y_val) containing the training and validation datasets.
    """
    X, y = [], []

    for _, _, files in os.walk(img_folder):
        for f in files:
            label = f.split('.')[0]
            extension = f.split('.')[-1]
            if extension.lower() == 'png':
                img, label = encode_single_sample(os.path.join(img_folder, f), label, crop)
                X.append(img)
                y.append(label)

    if len(y) == 0:
        raise ValueError("No PNG images found in the specified folder.")

    X = np.array(X)
    y = np.array(y)

    num_samples = X.shape[0]  

    # Split the dataset
    X_train, X_val, y_train, y_val = train_test_split(
        X.reshape(num_samples, -1), y, test_size=test_size, shuffle=True, random_state=42
    )

    # Reshape to original dimensions
    original_height, original_width, original_channels = X.shape[1], X.shape[2], X.shape[3]

    train_samples = X_train.shape[0]
    val_samples = X_val.shape[0]

    X_train = X_train.reshape(train_samples, original_height, original_width, original_channels)
    X_val = X_val.reshape(val_samples, original_height, original_width, original_channels)

    return X_train, X_val, y_train, y_val
