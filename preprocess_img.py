import cv2
import numpy as np

def preprocess_img(input_image, PREPROCESS_DIMS):
    preprocessed = cv2.resize(input_image, PREPROCESS_DIMS)
    #preprocessed = preprocessed - 127.5
    #preprocessed = preprocessed / 127.5
    preprocessed = preprocessed.astype(np.float32)

    return preprocessed