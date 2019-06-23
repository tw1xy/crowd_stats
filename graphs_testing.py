from mvnc import mvncapi as mvnc

import numpy as np
import cv2
import os
import time
import argparse

CLASSES = ("background", "aeroplane", "bicycle", "bird",
"boat", "bottle", "bus", "car", "cat", "chair", "cow",
"diningtable", "dog", "horse", "motorbike", "person",
"pottedplant", "sheep", "sofa", "train", "tvmonitor")

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

DIRECTORY = "images"
FILENAME = "1.jpg"

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--mode", required=True,
	help="running mode: 'image' for images; 'cam' for pc cam; 'rcam' for raspberry cam")
ap.add_argument("-f", "--file_name", help=" video file to run inferences on")
args = vars(ap.parse_args())

mode = args["mode"]
file_name = args["file_name"]

#cap = cv2.imread(os.path.join(DIRECTORY, FILENAME))	

PREPROCESS_DIMS = (300, 300)




GRAPH_FILEPATH1 = 'graphs/g_300'
#the next graph is from https://github.com/BeloborodovDS/MobilenetSSDFace
#GRAPH_FILEPATH2 = 'graphs/ssd-face-longrange'

def process_prediction(output):
    num_valid_boxes = int(output[0])
    predictions = []
    for box_index in range(num_valid_boxes):
        base_index = 7 + box_index * 7

        if (not np.isfinite(output[base_index]) or
            not np.isfinite(output[base_index + 1]) or
            not np.isfinite(output[base_index + 2]) or
            not np.isfinite(output[base_index + 3]) or
            not np.isfinite(output[base_index + 4]) or
            not np.isfinite(output[base_index + 5]) or
            not np.isfinite(output[base_index + 6])):
            continue
        
        (h, w) = img.shape[:2]
        x1 = max(0, int(output[base_index + 3] * w))
        y1 = max(0, int(output[base_index + 4] * h))
        x2 = min(w,	int(output[base_index + 5] * w))
        y2 = min(h,	int(output[base_index + 6] * h))
        
        pred_class = int(output[base_index + 1])
        pred_conf = output[base_index + 2]
        pred_boxpts = ((x1, y1), (x2, y2))
        prediction = (pred_class, pred_conf, pred_boxpts)
        predictions.append(prediction)
    return predictions

def draw_output(predictions,image_for_result):
    for (i, pred) in enumerate(predictions):
        (pred_class, pred_conf, pred_boxpts) = pred

        if pred_conf > 0.5:
            print("[INFO] Prediction #{}: class={}, confidence={}, "
                "boxpoints={}".format(i, CLASSES[pred_class], pred_conf,pred_boxpts))

            label = "{}: {:.2f}%".format(CLASSES[pred_class],
                pred_conf * 100)

            (ptA, ptB) = (pred_boxpts[0], pred_boxpts[1])
            ptA = (int(ptA[0] * DISP_MULTIPLIER_X), int(ptA[1] * DISP_MULTIPLIER_Y))
            ptB = (int(ptB[0] * DISP_MULTIPLIER_X), int(ptB[1] * DISP_MULTIPLIER_Y))
            (startX, startY) = (ptA[0], ptA[1])
            y = startY - 15 if startY - 15 > 15 else startY + 15

            cv2.rectangle(image_for_result, ptA, ptB,
                COLORS[pred_class], 2)
            cv2.putText(image_for_result, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[pred_class], 2)
    return image_for_result

def process_faces(predictions):
    for i in range():
        pass

def pre_process_img(img,PREPROCESS_DIMS):
    img = cv2.resize(img, PREPROCESS_DIMS)
    img = img - 127.5
    img = img / 127.5
    return img.astype(np.float32)


device_list = mvnc.enumerate_devices()
device = mvnc.Device(device_list[0])
device.open()

#Allocate first grahp
with open(GRAPH_FILEPATH1, mode='rb') as f: graph_buffer1 = f.read()
graph1 = mvnc.Graph('graph1')

input_fifo, output_fifo = graph1.allocate_with_fifos(device, graph_buffer1)

#cap = cv2.imread(os.path.join(DIRECTORY, FILENAME))

if mode == "cam":
	cap = cv2.VideoCapture(0)
	print("[INFO] starting the video stream from camera")

if mode == "vid":
    if file_name:
        cap = cv2.VideoCapture(file_name)
    else:
        cap = cv2.VideoCapture('video/1.mp4')
	
    print("[INFO] starting the video stream from video")
    

if mode == "rcam":
	cap = VideoStream(usePiCamera=True).start()
	print("[INFO] starting the video stream from raspberry")

z=0
ret, frame = cap.read()

DISPLAY_DIMS = (frame.shape[0], frame.shape[1])
DISP_MULTIPLIER_Y = DISPLAY_DIMS[0] / PREPROCESS_DIMS[0]
DISP_MULTIPLIER_X = DISPLAY_DIMS[1] / PREPROCESS_DIMS[1]


if mode == "cam" or mode == "rcam" or mode == "vid":
    while True:
        startT = time.clock()
        ret, frame = cap.read()
        image_for_result = frame
        img = pre_process_img(frame,PREPROCESS_DIMS)
        
        start = time.clock()
        graph1.queue_inference_with_fifo_elem(input_fifo, output_fifo, img, None)
        output, user_obj = output_fifo.read_elem()
        print("time: {} and fps: {} ".format(time.clock() - start,1/(time.clock() - start)))


        preds = process_prediction(output)

        img_out = draw_output(preds,image_for_result)

        cv2.imshow('image',img_out)
        if z==0:
            cv2.imwrite("images/1_v2.jpg",img_out)
            z=1

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        print("Atime: {} and Afps: {} ".format(time.clock() - startT,1/(time.clock() - startT)))

input_fifo.destroy()
output_fifo.destroy()
graph1.destroy()
device.close()
device.destroy()