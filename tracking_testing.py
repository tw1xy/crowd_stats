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
#cap = cv2.VideoCapture('images/benidorm1.mp4')
#cap = VideoStream(usePiCamera=True).start()
ret, frame = cap.read()

#for showing image
DISPLAY_DIMS = (frame.shape[0], frame.shape[1])

DISP_MULTIPLIER_Y = (DISPLAY_DIMS[0] / PREPROCESS_DIMS[0])
DISP_MULTIPLIER_X = (DISPLAY_DIMS[1] / PREPROCESS_DIMS[1])

DISP_MULT = DISP_MULTIPLIER_X, DISP_MULTIPLIER_Y

#Initializing tracker
ct = CentroidTracker()


while True:
    startT =time.clock()
    ret, frame = cap.read()
    

    image_for_result = frame
    img = sf.pre_process_img(frame,PREPROCESS_DIMS)

    #print("time_frame_aq: {:.0f} ms".format((time.clock()-startT)*1000))
    start =time.clock()

    graph.queue_inference_with_fifo_elem(input_fifo, output_fifo, img, None)
    output, user_obj = output_fifo.read_elem()

    #print("predicting: {:.0f} ms".format((time.clock()-start)*1000))

    predictions, boxes = sf.process_prediction(output,PREPROCESS_DIMS,DISP_MULT)

    img_out= sf.draw_output(predictions,image_for_result,DISP_MULT)
    
    #select only person boxes
    #boxes_p = sf.person_boxes_only(predictions)
    #TRACKING STARTS HERE
    people = ct.update(boxes)
    for (objectID, centroid) in people.items():
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (int(centroid[0] - 10), int(centroid[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (int(centroid[0]), int(centroid[1])), 4, (0, 255, 0), -1)


    cv2.imshow('image',img_out)

    #print("Total: {:.0f} ms".format((time.clock()-startT)*1000))

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cap.release()
input_fifo.destroy()
output_fifo.destroy()
graph.destroy()
device.close()
device.destroy()
