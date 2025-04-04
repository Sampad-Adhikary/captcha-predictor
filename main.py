import json
import numpy as np
import cv2
import tensorflow as tf
import os
import sys
from datetime import datetime
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# from my_encoding_module import decode_prediction  # You need to define this
from image_processor import process_image

# import files


def predict_captcha(img_path):
    """
    Given an image path, returns the predicted CAPTCHA string.
    """

    vocabulary = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
    char_to_num = {c:i for i,c in enumerate(vocabulary)}
    num_to_char = {str(v): k for k, v in char_to_num.items()}
    # print(num_to_char)

    # Load the trained model
    MODEL_PATH = "/home/invincibleocean/Work/captcha-predictor/models/captcha-breaker-1.keras"
    model = tf.keras.models.load_model(MODEL_PATH)

    img = process_image(img_path)
    y_pred = model.predict(img)
    y_pred = np.argmax(y_pred, axis=2)[0]  # Remove batch dimension
    # print(y_pred,type(y_pred))
    # for i in y_pred:
        
    return ''.join([num_to_char.get(str(i), '?') for i in y_pred])

image_paths = [f for f in os.listdir('/home/invincibleocean/Work/captcha-predictor/input_data') if os.path.isfile(os.path.join('/home/invincibleocean/Work/captcha-predictor/input_data', f))]

captcha_results = {}

for image_path  in image_paths:
    print('image path: ',image_path)
    start_time = datetime.now()
    try:
        predicted_text = predict_captcha(f'/home/invincibleocean/Work/captcha-predictor/input_data/{image_path}')
        captcha_results[image_path.split('.')[0]] = predicted_text
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
    print('Executed in {} seconds'.format((datetime.now() - start_time).total_seconds()))

print(captcha_results)

correct_count = 0
total_count = 0
for image_path, predicted_text in captcha_results.items():
    # print(f"Image: {image_path}, Predicted Text: {predicted_text}")
    if image_path == predicted_text:
        correct_count += 1
    total_count += 1

with open('results.json', 'w') as f:
    json.dump(captcha_results, f)

accuracy = correct_count / total_count
print('Accuracy: {:.2f}%'.format(accuracy * 100))   