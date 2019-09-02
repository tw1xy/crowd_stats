from mvnc import mvncapi as mvnc

import cv2
import os
import time
import numpy as np

from centroidtracker_mine import CentroidTracker
import support_functions as sf

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
ret, frame = cap.read()
print("Camera is ON.")

#IN ORDER TO DISPLAY IMAGE ON THE INPUT RESOLUTION
DISPLAY_DIMS = (frame.shape[0], frame.shape[1])
DISP_MULTIPLIER_Y_300 = (DISPLAY_DIMS[0] / PREPROCESS_DIMS_300[0])
DISP_MULTIPLIER_X_300 = (DISPLAY_DIMS[1] / PREPROCESS_DIMS_300[1])
DISP_MULT_300 = DISP_MULTIPLIER_X_300, DISP_MULTIPLIER_Y_300

#Initializing tracker
ct = CentroidTracker()

start_time = time.time()
FRAME_COUNTER = 0
while True:
    try:
        ret, frame = cap.read()
        FRAME_COUNTER = FRAME_COUNTER + 1
        
        image_for_result = frame.copy() 
        img = sf.pre_process_img(frame,PREPROCESS_DIMS_300)

        person_graph.queue_inference_with_fifo_elem(input_fifo_person,output_fifo_person,img,None)
        person_output, user_obj = output_fifo_person.read_elem()

        person_predictions, boxes_person = sf.process_prediction(person_output,PREPROCESS_DIMS_300,DISP_MULT_300)

        img_out= sf.draw_output(person_predictions,image_for_result,DISP_MULT_300)


        #TRACKING STARTS HERE
        people = ct.update(boxes_person)
        
        #check if all faces have age/gender
        for person in list(people.keys()):
            count = 0
            if ct.age(person):
                count += 1      

        if len(person_predictions) and not count == len(list(people.keys())):
            #then search faces
            #print("searching faces")
            #print("N_persons: {}; N_faces: {}; N_ages: {}".format(len(person_predictions),len(list(people.keys())),count))
            face_graph.queue_inference_with_fifo_elem(input_fifo_face,output_fifo_face,img,None)
            face_output, user_obj = output_fifo_face.read_elem()

            face_predictions, boxes_faces = sf.process_prediction(face_output,PREPROCESS_DIMS_300,DISP_MULT_300)
            img_out= sf.draw_output(face_predictions,image_for_result,DISP_MULT_300)

        for (objectID, centroid) in people.items():
            

            if ct.person(objectID) and ct.age(objectID) and ct.gender(objectID):
                person_is = ct.person(objectID)
            else:
                #print("Catching person_box for ID {}".format(objectID))
                for box in boxes_person:
                    if centroid[0] > box[0] and centroid[0] < box[2] and centroid[1] > box[1] and centroid[1] < box[3]:
                        ct.reg_person(objectID,box)
                    else:
                        #print("No person box for ID {}".format(objectID))
                        pass
            
            if ct.face(objectID) and ct.age(objectID) and ct.gender(objectID):
                ct.face(objectID)
            else:
                for boxx in boxes_faces:
                    centroid_face = (int((boxx[0] + boxx[2]) / 2.0),int((boxx[1] + boxx[3]) / 2.0))
                    if centroid_face[0] > box[0] and centroid_face[0] < box[2] and centroid_face[1] > box[1] and centroid_face[1] < box[3]:
                        ct.reg_face(objectID,boxx)
                    else:
                        #print("No face for ID {}".format(objectID))
                        pass



            if ct.age(objectID):
                age_is = ct.age(objectID)
            if not ct.age(objectID) and ct.face(objectID):
                
                box_is = ct.face(objectID)
                
                try:
                    cropped_image = frame[(box_is[1]-int(box_is[1]*PERCENTAGE)):(box_is[3]+int(box_is[3]*PERCENTAGE)),(box_is[0]-int(box_is[0]*PERCENTAGE)):(box_is[2]+int(box_is[2]*PERCENTAGE))]
                    cv2.imshow('face',cropped_image)
                except:
                    cropped_image = frame[(box_is[1]):(box_is[3]),(box_is[0]):(box_is[2])]
                    cv2.imshow('face',cropped_image)

                #pre-process img in order to predict age 
                face_img = sf.pre_process_img_2(cropped_image,PREPROCESS_DIMS_227,ilsvrc_mean)
                
                fc_img_c1 = face_img.copy()
                age_graph.queue_inference_with_fifo_elem(input_fifo_age,output_fifo_age,fc_img_c1,None)
                age_output, user_obj = output_fifo_age.read_elem()
                age,age_prob = sf.process_age(age_output)
            
                age_is = age
                ct.reg_age(objectID,age)

                fc_img_c2 = face_img.copy()
                gender_graph.queue_inference_with_fifo_elem(input_fifo_gender,output_fifo_gender,fc_img_c2,None)
                gender_output, user_obj = output_fifo_gender.read_elem()
                gender, gender_prob = sf.process_gender(gender_output)

                gender_is = gender
                ct.reg_gender(objectID,gender)
                print("ID:{} Age:{} Gender: {}".format(objectID, age_is, gender_is))
            
            if not ct.face(objectID):
                age_is = "None"
                gender_is = "None"
                

            #text = "ID:{} Age:{} Gender: {}".format(objectID, age_is, gender_is)
            text = "ID:{}".format(objectID)
            cv2.putText(img_out, text, (int(centroid[0] - 10), int(centroid[1] - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(img_out, (int(centroid[0]), int(centroid[1])), 4, (0, 255, 0), -1)
    

        cv2.imshow('image',img_out)
    except:
        print("There's some error in the main loop, please take a look")
        break
    #print("Total: {:.0f} ms".format((time.clock()-startT)*1000))

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

print("Done {} frames averaging {:.1f} FPS or {:.3f} seconds per frame. Total time: {:.1f} s".format(FRAME_COUNTER,FRAME_COUNTER/(time.time()-start_time),(time.time()-start_time)/FRAME_COUNTER,time.time()-start_time))

cap.release()

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
