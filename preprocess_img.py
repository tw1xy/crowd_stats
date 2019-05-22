import cv2
import numpy as np
ilsvrc_mean = np.load('ilsvrc_2012_mean.npy').mean(1).mean(1)

def preprocess_img(input_image, PREPROCESS_DIMS):
    preprocessed = cv2.resize(input_image, PREPROCESS_DIMS)
    #preprocessed = preprocessed - 127.5
    #preprocessed = preprocessed / 127.5
    img = preprocessed.astype(np.float32)
    img[:,:,0] = (img[:,:,0] - ilsvrc_mean[0])
    img[:,:,1] = (img[:,:,1] - ilsvrc_mean[1])
    img[:,:,2] = (img[:,:,2] - ilsvrc_mean[2])

    return img