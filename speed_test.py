from mvnc import mvncapi as mvnc

import cv2
import os
import time
import numpy as np
import argparse

from scipy.spatial import distance as dist
from collections import OrderedDict

from pyimagesearch.centroidtracker_mine import CentroidTracker
import data as Data_support
import support_functions as sf

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mode", required=False,
	help="running mode: 'NCS' to run NCS")
ap.add_argument("-t", "--test_name", help=" test to preform, can be: 'cap_time'; ")
ap.add_argument("-f", "--frames_limit", required=True, help="frames limit")
args = vars(ap.parse_args())

mode = args["mode"]
test = args["test_name"]
frames_limit = args["frames_limit"]

PERSON_GRAPH = 'graphs/g_300' #300
FACE_GRAPH = 'graphs/ssd-face' #300
AGE_GRAPH = 'graphs/age_model' #227 +mean
GENDER_GRAPH = 'graphs/gender_model' #227 +mean

AGE_GRAPH_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
GENDER_GRAPH_LIST = ['Male','Female']

PREPROCESS_DIMS_300 = (300, 300)
PREPROCESS_DIMS_227 = (227, 227)

PERCENTAGE = 0.05

ilsvrc_mean = np.load('graphs/age_gender_mean.npy').mean(1).mean(1) #loading the mean file

if mode == "NCS":
    #OPEN DEVICE
    device_list = mvnc.enumerate_devices()
    if device_list:
        device = mvnc.Device(device_list[0])
        device.open()
        print("Found {} device(s) and connected to {}.".format(len(device_list),device))
    else:
        print("0 devices found, please make sure the NCS is connected. Exiting...")
        os.sys.exit()

    #ALLOCATING GRAPHS
    with open(PERSON_GRAPH, mode='rb') as f: person_graph_buffer = f.read()
    person_graph = mvnc.Graph('person_graph')
    input_fifo_person, output_fifo_person = person_graph.allocate_with_fifos(device, person_graph_buffer)

    with open(FACE_GRAPH, mode='rb') as f: face_graph_buffer = f.read()
    face_graph = mvnc.Graph('face_graph')
    input_fifo_face, output_fifo_face = face_graph.allocate_with_fifos(device, face_graph_buffer)

    with open(AGE_GRAPH, mode='rb') as f: age_graph_buffer = f.read()
    age_graph = mvnc.Graph('age_graph')
    input_fifo_age, output_fifo_age = age_graph.allocate_with_fifos(device, age_graph_buffer)

    with open(GENDER_GRAPH, mode='rb') as f: gender_graph_buffer = f.read()
    gender_graph = mvnc.Graph('gender_graph')
    input_fifo_gender, output_fifo_gender = gender_graph.allocate_with_fifos(device, gender_graph_buffer)

    print("Graphs allocated.")

cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("images/10min18.mp4")
ret, frame = cap.read()

print("Camera is ON.")

#IN ORDER TO DISPLAY IMAGE ON THE INPUT RESOLUTION
DISPLAY_DIMS = (frame.shape[0], frame.shape[1])
DISP_MULTIPLIER_Y_300 = (DISPLAY_DIMS[0] / PREPROCESS_DIMS_300[0])
DISP_MULTIPLIER_X_300 = (DISPLAY_DIMS[1] / PREPROCESS_DIMS_300[1])
DISP_MULT_300 = DISP_MULTIPLIER_X_300, DISP_MULTIPLIER_Y_300

#Initializing tracker
ct = CentroidTracker()

captur_time = []
processing_time = []
inf1_time = []
proc_inference_time = []

start_time = time.time()
FRAME_COUNTER = 0

while True:
    time_0 = time.time()
    ret, frame = cap.read()
    time_00 = time.time()
    FRAME_COUNTER += 1

    image_for_result = frame.copy() 
    time_1 = time.time()
    img = sf.pre_process_img(frame,PREPROCESS_DIMS_300)
    time_2 = time.time()

    time_3 = time.time()
    person_graph.queue_inference_with_fifo_elem(input_fifo_person,output_fifo_person,img,None)
    person_output, user_obj = output_fifo_person.read_elem()
    #print(person_output[0])
    time_4 = time.time()

    time_5 = time.time()
    person_predictions, boxes_person = sf.process_prediction(person_output,PREPROCESS_DIMS_300,DISP_MULT_300)
    time_6 = time.time()










    #time.sleep(0.1)
    captur_time.append(time_00 - time_0)
    processing_time.append(time_2 - time_1)
    inf1_time.append(time_4 - time_3)
    proc_inference_time.append(time_6 - time_5)



    if FRAME_COUNTER == int(frames_limit):
        break

print("Done {} frames in {:.3f} seconds or {:.1f} FPS, {:.3f} seconds per image".format(FRAME_COUNTER,(time.time()-start_time),FRAME_COUNTER/(time.time()-start_time),(time.time()-start_time)/FRAME_COUNTER))

captur_time_ = sum(captur_time,0)/len(captur_time)
print("Capturing time: {:.3f}".format(captur_time_))

proc_time_ = sum(processing_time,0)/len(processing_time)
print("Processig time: {:.3f}".format(proc_time_))

inf1_time_ = sum(inf1_time,0)/len(inf1_time)
print("Inferencing time: {:.3f}".format(inf1_time_))

proc_inference_time_ = sum(proc_inference_time,0)/len(proc_inference_time)
print("Preocessing inference time: {:.3f}".format(proc_inference_time_))





cap.release()

if mode == "NCS":
    input_fifo_person.destroy()
    output_fifo_person.destroy()
    person_graph.destroy()
    input_fifo_face.destroy()
    output_fifo_face.destroy()
    face_graph.destroy()
    input_fifo_age.destroy()
    output_fifo_age.destroy()
    age_graph.destroy()
    input_fifo_gender.destroy()
    output_fifo_gender.destroy()
    gender_graph.destroy()

    device.close()
    device.destroy()