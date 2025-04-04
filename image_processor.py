import cv2
import numpy as np
import os

def process_image(image_path):
    # Read image in unchanged mode
    captcha_img_rgba = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    # Extract alpha channel (if available) or convert to grayscale
    if captcha_img_rgba.shape[2] == 4:
        print('Alpha channel avaliable')
        alpha_channel = captcha_img_rgba[:, :, 3]
        text_mask = cv2.bitwise_not(alpha_channel)
    else:
        text_mask = cv2.cvtColor(captcha_img_rgba, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to binarize the image
    _, text_thresh = cv2.threshold(text_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Detect horizontal lines
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))  # Adjust kernel size based on line thickness
    horizontal_lines = cv2.morphologyEx(text_thresh, cv2.MORPH_OPEN, kernel, iterations=2)

    # Subtract detected lines from the original thresholded image
    text_cleaned = cv2.bitwise_or(text_thresh, horizontal_lines)


    kernel_thin = np.ones((1, 1), np.uint8)  # Increase height of erosion
    text_thinned = cv2.erode(text_cleaned, kernel_thin, iterations=30)  # More aggressive erosion

    kernel_thin = np.ones((1, 1), np.uint8)  # Increase height of dilation
    text_thinned = cv2.dilate(text_thinned, kernel_thin, iterations=30)

    target_width = 200
    target_height = 50

    # Resize the width to 200 while keeping height the same
    resized = cv2.resize(text_thinned, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    # Print shape of the resized image
    print(f"Resized image shape: {resized.shape}")
    # save image
    
    
    processed = resized.astype("float32") / 255.0             # Normalize
    processed = np.expand_dims(processed, axis=-1)            # (50, 200, 1)
    processed = np.transpose(processed, (1, 0, 2))            # (200, 50, 1)
    processed = np.expand_dims(processed, axis=0)   
    
    # cv2.imwrite(os.path.join(os.path.dirname('new.png'), 'resized_' + os.path.basename(image_path)), processed)
    # cv2.imwrite(os.path.join(os.path.dirname(image_path), 'resized_' + os.path.basename(image_path)), processed[0].transpose(1, 0, 2) * 255.0)

    # print(processed.shape)
    return processed

