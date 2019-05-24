from mvnc import mvncapi as mvnc

import numpy as np
import cv2
import os
import time

CLASSES = ("background", "aeroplane", "bicycle", "bird",
"boat", "bottle", "bus", "car", "cat", "chair", "cow",
"diningtable", "dog", "horse", "motorbike", "person",
"pottedplant", "sheep", "sofa", "train", "tvmonitor")

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

DIRECTORY = "images"
FILENAME = "1.jpg"

#cap = cv2.imread(os.path.join(DIRECTORY, FILENAME))	


PREPROCESS_DIMS = (300, 300)
DISPLAY_DIMS = (500, 500)
DISP_MULTIPLIER_Y = DISPLAY_DIMS[0] // PREPROCESS_DIMS[0]
DISP_MULTIPLIER_X = DISPLAY_DIMS[1] // PREPROCESS_DIMS[1]


GRAPH_FILEPATH1 = 'graphs/g_s12'
#the next graph is from https://github.com/BeloborodovDS/MobilenetSSDFace
GRAPH_FILEPATH2 = 'graphs/ssd-face-longrange'

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
            ptA = (ptA[0] * DISP_MULTIPLIER_X, ptA[1] * DISP_MULTIPLIER_Y)
            ptB = (ptB[0] * DISP_MULTIPLIER_X, ptB[1] * DISP_MULTIPLIER_Y)
            (startX, startY) = (ptA[0], ptA[1])
            y = startY - 15 if startY - 15 > 15 else startY + 15

            cv2.rectangle(image_for_result, ptA, ptB,
                COLORS[pred_class], 2)
            cv2.putText(image_for_result, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 2, COLORS[pred_class], 10)
    return image_for_result

def process_faces(img,predictions):
    outputs = []
    for (i, pred) in enumerate(predictions):
        (_, pred_conf, pred_boxpts) = pred
    
        if pred_conf > 0.5:
            (ptA, ptB) = (pred_boxpts[0], pred_boxpts[1])
            x = ptA[0]*DISP_MULTIPLIER_X
            y = ptA[1]*DISP_MULTIPLIER_Y
            xf = ptB[0]*DISP_MULTIPLIER_X
            yf = ptB[1]*DISP_MULTIPLIER_Y
            img_cropped = img[y:yf, x:xf]
            img_cropped = pre_process_img(img_cropped,PREPROCESS_DIMS)
        graph2.queue_inference_with_fifo_elem(input_fifo2, output_fifo2, img_cropped, None)
        output2, user_obj = output_fifo2.read_elem()
        outputs.append(output2)
    return outputs

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

#Allocate second grahp
with open(GRAPH_FILEPATH2, mode='rb') as f: graph_buffer2 = f.read()
graph2 = mvnc.Graph('graph2')

input_fifo2, output_fifo2 = graph2.allocate_with_fifos(device, graph_buffer2)

#first image input
#img = pre_process_img(image_for_result,PREPROCESS_DIMS)



# start = time.clock()
# graph2.queue_inference_with_fifo_elem(input_fifo2, output_fifo2, img, None)
# output2, user_obj = output_fifo2.read_elem()
# print(time.clock() - start)



#outputs = process_faces(cap,preds)
#preds2 = process_prediction(outputs[0])


#img_out = draw_output(preds2,img_out)


#cap = cv2.VideoCapture(0)
cap = cv2.imread(os.path.join(DIRECTORY, FILENAME))
PREPROCESS_DIMS = (300, 300)
#DISPLAY_DIMS = (cap.shape[0], cap.shape[1])
DISPLAY_DIMS = (600, 600)
DISP_MULTIPLIER_Y = DISPLAY_DIMS[0] // PREPROCESS_DIMS[0]
DISP_MULTIPLIER_X = DISPLAY_DIMS[1] // PREPROCESS_DIMS[1]
z=0
while True:

    #ret, frame = cap.read()
    frame = cap
    image_for_result = cv2.resize(frame,(DISPLAY_DIMS[1],DISPLAY_DIMS[0]))
    img = pre_process_img(frame,PREPROCESS_DIMS)
    
    start = time.clock()
    graph1.queue_inference_with_fifo_elem(input_fifo, output_fifo, img, None)
    output, user_obj = output_fifo.read_elem()
    print(time.clock() - start)

    preds = process_prediction(output)

    img_out = draw_output(preds,image_for_result)

    #cv2.imshow('image',img_out)
    if z==0:
        cv2.imwrite("images/1_v2.jpg",img_out)
        z=1

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

input_fifo.destroy()
output_fifo.destroy()
input_fifo2.destroy()
output_fifo2.destroy()
graph1.destroy()
graph2.destroy()
device.close()
device.destroy()