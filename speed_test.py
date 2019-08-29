from mvnc import mvncapi as mvnc

import cv2
import os
import time
import numpy as np
import argparse

from scipy.spatial import distance as dist
from collections import OrderedDict

from pyimagesearch.centroidtracker_mine import CentroidTracker
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

#cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("images/10min18.mp4")

#ret, frame = cap.read()

frame = cv2.imread('images/5.jpg')

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
drawi_time = []
track_time = []
face_time = []
proc_inference_time_2 = []
catch_box_for_ID_time = []
catch_face_for_ID_time = []
crop_and_process_time = []
age_inference_time = []
age_process_time = []
gender_inference_time = []
gender_process_time = []
drawi_time2 = []

start_time = time.time()
FRAME_COUNTER = 0

while True:
    time_0 = time.time()
    #ret, frame = cap.read()
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

    time_7 = time.time()
    img_out= sf.draw_output(person_predictions,image_for_result,DISP_MULT_300)
    time_8 = time.time()

    #TRACKING STARTS HERE
    time_9 = time.time()
    people = ct.update(boxes_person)
    time_10 = time.time()

    #check if all faces have age/gender
    #for person in list(people.keys()):
    #    count = 0
    #    if ct.age(person):
    #        count += 1      

    if len(person_predictions): #and not count == len(list(people.keys())):
        #then search faces
        #print("searching faces")
        #print("N_persons: {}; N_faces: {}; N_ages: {}".format(len(person_predictions),len(list(people.keys())),count))
        time_11 = time.time()
        face_graph.queue_inference_with_fifo_elem(input_fifo_face,output_fifo_face,img,None)
        face_output, user_obj = output_fifo_face.read_elem()
        time_12 = time.time()

        time_13 = time.time()
        face_predictions, boxes_faces = sf.process_prediction(face_output,PREPROCESS_DIMS_300,DISP_MULT_300)
        time_14 = time.time()
        img_out= sf.draw_output(face_predictions,image_for_result,DISP_MULT_300)

    for (objectID, centroid) in people.items():
        

        if ct.person(objectID) and ct.age(objectID) and ct.gender(objectID) and False:
            person_is = ct.person(objectID)
        else:
            #print("Catching person_box for ID {}".format(objectID))
            time_15 = time.time()
            for box in boxes_person:
                if centroid[0] > box[0] and centroid[0] < box[2] and centroid[1] > box[1] and centroid[1] < box[3]:
                    ct.reg_person(objectID,box)
                else:
                    #print("No person box for ID {}".format(objectID))
                    pass
            time_16 = time.time()


        if ct.face(objectID) and ct.age(objectID) and ct.gender(objectID) and False:
            ct.face(objectID)
        else:
            time_17 = time.time()
            for boxx in boxes_faces:
                centroid_face = (int((boxx[0] + boxx[2]) / 2.0),int((boxx[1] + boxx[3]) / 2.0))
                if centroid_face[0] > box[0] and centroid_face[0] < box[2] and centroid_face[1] > box[1] and centroid_face[1] < box[3]:
                    ct.reg_face(objectID,boxx)
                else:
                    #print("No face for ID {}".format(objectID))
                    pass
            time_18 = time.time()
            

        if ct.face(objectID):
            time_19 = time.time()
            box_is = ct.face(objectID)
            
            try:
                cropped_image = frame[(box_is[1]-int(box_is[1]*PERCENTAGE)):(box_is[3]+int(box_is[3]*PERCENTAGE)),(box_is[0]-int(box_is[0]*PERCENTAGE)):(box_is[2]+int(box_is[2]*PERCENTAGE))]
                #cv2.imshow('face',cropped_image)
            except:
                cropped_image = frame[(box_is[1]):(box_is[3]),(box_is[0]):(box_is[2])]
                #cv2.imshow('face',cropped_image)

            #pre-process img in order to predict age 
            face_img = sf.pre_process_img_2(cropped_image,PREPROCESS_DIMS_227,ilsvrc_mean)
            
            fc_img_c1 = face_img.copy()

            time_20 = time.time()

            
            age_graph.queue_inference_with_fifo_elem(input_fifo_age,output_fifo_age,fc_img_c1,None)
            age_output, user_obj = output_fifo_age.read_elem()

            time_21 = time.time()

            age,age_prob = sf.process_age(age_output)

            time_22 = time.time()
            
            
            age_is = age
            ct.reg_age(objectID,age)

            fc_img_c2 = face_img.copy()

            time_23 = time.time()
            gender_graph.queue_inference_with_fifo_elem(input_fifo_gender,output_fifo_gender,fc_img_c2,None)
            gender_output, user_obj = output_fifo_gender.read_elem()
            time_24 = time.time()

            gender, gender_prob = sf.process_gender(gender_output)
            time_25 = time.time()

            gender_is = gender
            ct.reg_gender(objectID,gender)
            #print("ID:{} Age:{} Gender: {}".format(objectID, age_is, gender_is))
        
        if not ct.face(objectID):
            age_is = "None"
            gender_is = "None"
            
        time_26 = time.time()
        #text = "ID:{} Age:{} Gender: {}".format(objectID, age_is, gender_is)
        text = "ID:{}".format(objectID)
        cv2.putText(img_out, text, (int(centroid[0] - 10), int(centroid[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(img_out, (int(centroid[0]), int(centroid[1])), 4, (0, 255, 0), -1)

        time_27 = time.time()


        catch_box_for_ID_time.append(time_16 - time_15)
        catch_face_for_ID_time.append(time_18 - time_17)
        crop_and_process_time.append(time_20 - time_19)
        age_inference_time.append(time_21 - time_20)
        age_process_time.append(time_22 - time_21)
        gender_inference_time.append(time_24 - time_23)
        gender_process_time.append(time_25 - time_24)
        drawi_time2.append(time_27 - time_26)


    cv2.imshow('image',img_out)



    #time.sleep(0.1)
    captur_time.append(time_00 - time_0)
    processing_time.append(time_2 - time_1)
    inf1_time.append(time_4 - time_3)
    proc_inference_time.append(time_6 - time_5)
    drawi_time.append(time_8 - time_7)
    track_time.append(time_10 - time_9)
    face_time.append(time_12 - time_11)
    proc_inference_time_2.append(time_14 - time_13)

    key = cv2.waitKey(1) & 0xFF
    if FRAME_COUNTER == int(frames_limit) or key == ord("q"):
        break

print("Done {} frames in {:.3f} seconds or {:.1f} FPS, {:.1f} ms per image".format(FRAME_COUNTER,(time.time()-start_time),FRAME_COUNTER/(time.time()-start_time),(1000*(time.time()-start_time)/FRAME_COUNTER)))

captur_time_ = sum(captur_time,0)/len(captur_time)
print("Capturing time: {:.1f} ms".format(captur_time_*1000))

proc_time_ = sum(processing_time,0)/len(processing_time)
print("Processig time(Rsize and normalization): {:.1f} ms".format(proc_time_*1000))

inf1_time_ = sum(inf1_time,0)/len(inf1_time)
print("People inference time: {:.1f} ms".format(inf1_time_*1000))

proc_inference_time_ = sum(proc_inference_time,0)/len(proc_inference_time)
print("Preocessing inference time: {:.1f} ms".format(proc_inference_time_*1000))

drawi_time_ = sum(drawi_time,0)/len(drawi_time)
print("Drawing time_1: {:.1f} ms".format(drawi_time_*1000))

track_time_ = sum(track_time,0)/len(track_time)
print("Tracking time: {:.1f} ms".format(track_time_*1000))

face_time_ = sum(face_time,0)/len(face_time)
print("Face inference time: {:.1f} ms".format(face_time_*1000))

proc_inference_time_2_ = sum(proc_inference_time_2,0)/len(proc_inference_time_2)
print("Processig inference time 2: {:.1f} ms".format(proc_inference_time_2_*1000))

catch_box_for_ID_time_ = sum(catch_box_for_ID_time,0)/len(catch_box_for_ID_time)
print("Catch box for ID time: {:.1f} ms".format(catch_box_for_ID_time_*1000))

catch_face_for_ID_time_ = sum(catch_face_for_ID_time,0)/len(catch_face_for_ID_time)
print("Catch face for ID time: {:.1f} ms".format(catch_face_for_ID_time_*1000))

crop_and_process_time_ = sum(crop_and_process_time,0)/len(crop_and_process_time)
print("Crop face and process time: {:.1f} ms".format(crop_and_process_time_*1000))

age_inference_time_ = sum(age_inference_time,0)/len(age_inference_time)
print("Age inference time: {:.1f} ms".format(age_inference_time_*1000))

age_process_time_ = sum(age_process_time,0)/len(age_process_time)
print("Age process time: {:.1f} ms".format(age_process_time_*1000))

gender_inference_time_ = sum(gender_inference_time,0)/len(gender_inference_time)
print("Gender inference time: {:.1f} ms".format(gender_inference_time_*1000))

gender_process_time_ = sum(gender_process_time,0)/len(gender_process_time)
print("Gender process time: {:.1f} ms".format(gender_process_time_*1000))

drawi_time2_ = sum(drawi_time2,0)/len(drawi_time2)
print("Last drawing time: {:.1f} ms".format(drawi_time2_*1000))

#cap.release()

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