from mvnc import mvncapi as mvnc

import cv2
import os
import time

from scipy.spatial import distance as dist
from collections import OrderedDict

from pyimagesearch.centroidtracker import CentroidTracker
import support_functions as sf

PREPROCESS_DIMS = (300, 300)

GRAPH_FILEPATH = 'graphs/g_300'

device_list = mvnc.enumerate_devices()
device = mvnc.Device(device_list[0])
device.open()

with open(GRAPH_FILEPATH, mode='rb') as f: graph_buffer = f.read()
graph = mvnc.Graph('graph')

input_fifo, output_fifo = graph.allocate_with_fifos(device, graph_buffer)

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture('images/static1.mp4')
#cap = VideoStream(usePiCamera=True).start()
ret, frame = cap.read()

#for showing image
DISPLAY_DIMS = (frame.shape[0], frame.shape[1])

DISP_MULTIPLIER_Y = (DISPLAY_DIMS[0] / PREPROCESS_DIMS[0])
DISP_MULTIPLIER_X = (DISPLAY_DIMS[1] / PREPROCESS_DIMS[1])

DISP_MULT = DISP_MULTIPLIER_X, DISP_MULTIPLIER_Y


#Initializing tracker
tracker = cv2.TrackerGOTURN_create()
noboxes = 1


while True:
    ret, frame = cap.read()

    image_for_result = frame
    img = sf.pre_process_img(frame,PREPROCESS_DIMS)

    if noboxes:
        graph.queue_inference_with_fifo_elem(input_fifo, output_fifo, img, None)
        output, user_obj = output_fifo.read_elem()
        
        predictions, boxes = sf.process_prediction(output,PREPROCESS_DIMS,DISP_MULT)
        #img_out= sf.draw_output(predictions,image_for_result,DISP_MULT)
        print("Searching {}".format(time.clock()))
        if boxes:
            in_box = boxes[0]
            ok = tracker.init(image_for_result, in_box)
            noboxes = 0

    
    
    ok, in_box = tracker.update(image_for_result)
    #print(ok,in_box)
    if ok:
        print("Tracking{}".format(time.clock()))
        p1 = (int(in_box[0]), int(in_box[1]))
        p2 = (int(in_box[0] + in_box[2]), int(in_box[1] + in_box[3]))
        cv2.rectangle(image_for_result, p1, p2, (255,0,0), 2, 1)
        noboxes = 0

    if not ok:
        noboxes = 1

    cv2.imshow("Tracking", image_for_result)

    k = cv2.waitKey(1) & 0xff
    if k == ord("q") : break

cap.release()
input_fifo.destroy()
output_fifo.destroy()
graph.destroy()
device.close()
device.destroy()