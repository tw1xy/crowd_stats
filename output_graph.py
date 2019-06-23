from mvnc import mvncapi as mvnc

import numpy as np
import cv2

def pre_process_img(img,PREPROCESS_DIMS):
    img = cv2.resize(img, PREPROCESS_DIMS)
    img = img - 127.5
    img = img / 127.5
    return img.astype(np.float32)

#path to the graph file
GRAPH_FILEPATH1 = 'graphs/graph2'

device_list = mvnc.enumerate_devices()
device = mvnc.Device(device_list[0])
device.open()

with open(GRAPH_FILEPATH1, mode='rb') as f: graph_buffer1 = f.read()
graph1 = mvnc.Graph('graph1')

input_fifo, output_fifo = graph1.allocate_with_fifos(device, graph_buffer1)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()

PREPROCESS_DIMS = (300, 300)

img = pre_process_img(frame,PREPROCESS_DIMS)

graph1.queue_inference_with_fifo_elem(input_fifo, output_fifo, img, None)
output, user_obj = output_fifo.read_elem()

print("shape is: {}".format(np.shape(output)))

input_fifo.destroy()
output_fifo.destroy()
graph1.destroy()
device.close()
device.destroy()