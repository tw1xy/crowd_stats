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

cap = cv2.imread(os.path.join(DIRECTORY, FILENAME))	
image_for_result = cap.copy()

PREPROCESS_DIMS = (300, 300)
DISPLAY_DIMS = (cap.shape[0], cap.shape[1])
DISP_MULTIPLIER_Y = DISPLAY_DIMS[0] // PREPROCESS_DIMS[0]
DISP_MULTIPLIER_X = DISPLAY_DIMS[1] // PREPROCESS_DIMS[1]
GRAPH_FILEPATH = 'graphs/g_s12'


device_list = mvnc.enumerate_devices()
device = mvnc.Device(device_list[0])
device.open()

with open(GRAPH_FILEPATH, mode='rb') as f: graph_buffer = f.read()
graph = mvnc.Graph('graph1')

input_fifo, output_fifo = graph.allocate_with_fifos(device, graph_buffer)


img = cv2.resize(cap, PREPROCESS_DIMS)
img = img - 127.5
img = img / 127.5
img = img.astype(np.float32)
start = time.clock()
graph.queue_inference_with_fifo_elem(input_fifo, output_fifo, img, None)
output, user_obj = output_fifo.read_elem()
print(time.clock() - start)
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
            COLORS[pred_class], 10)
        cv2.putText(image_for_result, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 3, COLORS[pred_class], 10)

cv2.imwrite("images/1_v2.jpg",image_for_result)
