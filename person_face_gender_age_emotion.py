from mvnc import mvncapi as mvnc

import numpy as np
import cv2
import os
import time
import sys

import support_functions as sf

PREPROCESS_DIMS = (300, 300)

PERSON_GRAPH = 'graphs/g_300' #300
FACE_GRAPH = 'graphs/ssd-face' #300
AGE_GRAPH = 'graphs/age_model' #227 +mean
GENDER_GRAPH = 'graphs/gender_model' #227 +mean

AGE_GRAPH_LIST = ['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
GENDER_GRAPH_LIST = ['Male','Female']

PREPROCESS_DIMS_300 = (300, 300)
PREPROCESS_DIMS_227 = (227, 227)

ilsvrc_mean = np.load('graphs/age_gender_mean.npy').mean(1).mean(1) #loading the mean file

#OPEN DEVICE
device_list = mvnc.enumerate_devices()
if device_list:
    device = mvnc.Device(device_list[0])
    device.open()
    print("Found {} device(s) and connected to {}.".format(len(device_list),device))
else:
    print("0 devices found, please make sure the NCS is connected. Exiting...")
    sys.exit()

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
ret, frame = cap.read()
print("Camera is ON.")

#IN ORDER TO DISPLAY IMAGE ON THE INPUT RESOLUTION
DISPLAY_DIMS = (frame.shape[0], frame.shape[1])
DISP_MULTIPLIER_Y_300 = (DISPLAY_DIMS[0] / PREPROCESS_DIMS_300[0])
DISP_MULTIPLIER_X_300 = (DISPLAY_DIMS[1] / PREPROCESS_DIMS_300[1])
DISP_MULT_300 = DISP_MULTIPLIER_X_300, DISP_MULTIPLIER_Y_300

start_time = time.time()
FRAME_COUNTER = 0
while True:
    ret, frame = cap.read()
    FRAME_COUNTER = FRAME_COUNTER + 1
    

    image_for_result = frame.copy() 
    img = sf.pre_process_img(frame,PREPROCESS_DIMS)

    person_graph.queue_inference_with_fifo_elem(input_fifo_person,output_fifo_person,img,None)
    person_output, user_obj = output_fifo_person.read_elem()

    person_predictions, boxes = sf.process_prediction(person_output,PREPROCESS_DIMS_300,DISP_MULT_300)
    #print("Found {} people.".format(len(person_predictions)))
    img_out= sf.draw_output(person_predictions,image_for_result,DISP_MULT_300)
    #print(person_predictions[0][2])
    if len(person_predictions):
        #then serach faces
        face_graph.queue_inference_with_fifo_elem(input_fifo_face,output_fifo_face,img,None)
        face_output, user_obj = output_fifo_face.read_elem()

        face_predictions, boxes = sf.process_prediction(face_output,PREPROCESS_DIMS_300,DISP_MULT_300)
        #print("Found {} face(s).".format(len(face_predictions)))
        img_out = sf.draw_output(face_predictions,image_for_result,DISP_MULT_300)
        if boxes:
            box = boxes[0]

            cropped_image = frame[(box[1]):(box[3]),box[0]:box[2]]
            
            face_img = sf.pre_process_img_2(cropped_image,PREPROCESS_DIMS_227,ilsvrc_mean)
                   
            age_graph.queue_inference_with_fifo_elem(input_fifo_age,output_fifo_age,face_img,None)
            age_output, user_obj = output_fifo_age.read_elem()
            age = sf.process_age(age_output)

            gender_graph.queue_inference_with_fifo_elem(input_fifo_gender,output_fifo_gender,face_img,None)
            gender_output, user_obj = output_fifo_gender.read_elem()
            gender = sf.process_gender(gender_output)


            print("Age is {} and it's {} ".format(age,gender))
 
    
    cv2.imshow('image',img_out)




    key = cv2.waitKey(1) & 0xFF
    if key == ord("q") or FRAME_COUNTER == 500:
        break


print("Done {} frames averaging {:.1f} FPS or {:.3f} seconds per frame. Total time: {:.1f} s".format(FRAME_COUNTER,FRAME_COUNTER/(time.time()-start_time),(time.time()-start_time)/FRAME_COUNTER,time.time()-start_time))
#destroy all to close
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