from mvnc import mvncapi as mvnc
import tracker

import cv2
import os
import time


import support_functions as sf


CLASSES = ("background", "aeroplane", "bicycle", "bird",
"boat", "bottle", "bus", "car", "cat", "chair", "cow",
"diningtable", "dog", "horse", "motorbike", "person",
"pottedplant", "sheep", "sofa", "train", "tvmonitor")



PREPROCESS_DIMS = (300, 300)

GRAPH_FILEPATH = 'graphs/ssd-face'

device_list = mvnc.enumerate_devices()
device = mvnc.Device(device_list[0])
device.open()

with open(GRAPH_FILEPATH, mode='rb') as f: graph_buffer = f.read()
graph = mvnc.Graph('graph')

input_fifo, output_fifo = graph.allocate_with_fifos(device, graph_buffer)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

#for showing image
DISPLAY_DIMS = (frame.shape[0], frame.shape[1])

DISP_MULTIPLIER_Y = DISPLAY_DIMS[0] / PREPROCESS_DIMS[0]
DISP_MULTIPLIER_X = DISPLAY_DIMS[1] / PREPROCESS_DIMS[1]

DISP_MULT = DISP_MULTIPLIER_X, DISP_MULTIPLIER_Y

while True:
    ret, frame = cap.read()

    image_for_result = frame
    img = sf.pre_process_img(frame,PREPROCESS_DIMS)

    graph.queue_inference_with_fifo_elem(input_fifo, output_fifo, img, None)
    output, user_obj = output_fifo.read_elem()

    predictions = sf.process_prediction(output,PREPROCESS_DIMS)

    img_out= sf.draw_output(predictions,image_for_result,DISP_MULT)
    cv2.imshow('image',img_out)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

input_fifo.destroy()
output_fifo.destroy()
graph1.destroy()
device.close()
device.destroy()
