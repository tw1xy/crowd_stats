import time
from mvnc import mvncapi as mvnc

import numpy as np
import cv2
#import os

import preprocess_img as pimg


CLASSES = ("background", "aeroplane", "bicycle", "bird",
"boat", "bottle", "bus", "car", "cat", "chair", "cow",
"diningtable", "dog", "horse", "motorbike", "person",
"pottedplant", "sheep", "sofa", "train", "tvmonitor")

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

PREPROCESS_DIMS = (300, 300)
DISPLAY_DIMS = (600, 600)
DISP_MULTIPLIER = DISPLAY_DIMS[0] // PREPROCESS_DIMS[0]

GRAPH_FILEPATH = 'graphs/g_s12'

def process_data(image,input):
	output = input

	# grab the number of valid object predictions from the output,
	# then initialize the list of predictions
	num_valid_boxes = int(output[0])
	predictions = []

	# loop over results
	for box_index in range(num_valid_boxes):
		# calculate the base index into our array so we can extract
		# bounding box information
		base_index = 7 + box_index * 7

		# boxes with non-finite (inf, nan, etc) numbers must be ignored
		if (not np.isfinite(output[base_index]) or
			not np.isfinite(output[base_index + 1]) or
			not np.isfinite(output[base_index + 2]) or
			not np.isfinite(output[base_index + 3]) or
			not np.isfinite(output[base_index + 4]) or
			not np.isfinite(output[base_index + 5]) or
			not np.isfinite(output[base_index + 6])):
			continue

		# extract the image width and height and clip the boxes to the
		# image size in case network returns boxes outside of the image
		# boundaries
		(h, w) = image.shape[:2]
		x1 = max(0, int(output[base_index + 3] * w))
		y1 = max(0, int(output[base_index + 4] * h))
		x2 = min(w,	int(output[base_index + 5] * w))
		y2 = min(h,	int(output[base_index + 6] * h))

		# grab the prediction class label, confidence (i.e., probability),
		# and bounding box (x, y)-coordinates
		pred_class = int(output[base_index + 1])
		pred_conf = output[base_index + 2]
		pred_boxpts = ((x1, y1), (x2, y2))

		# create prediciton tuple and append the prediction to the
		# predictions list
		prediction = (pred_class, pred_conf, pred_boxpts)
		predictions.append(prediction)

	# return the list of predictions to the calling function
	return predictions




device_list = mvnc.enumerate_devices()
device = mvnc.Device(device_list[0])
device.open()


with open(GRAPH_FILEPATH, mode='rb') as f:
    graph_buffer = f.read()
graph = mvnc.Graph('graph1')

# Allocate the graph to the device
input_fifo, output_fifo = graph.allocate_with_fifos(device, graph_buffer)


cap = cv2.VideoCapture(0)

while True:
	try:
		ret, frame = cap.read()
		image_for_result = frame.copy()
		image_for_result = cv2.resize(image_for_result, DISPLAY_DIMS)
		
		input_tensor = pimg.preprocess_img(frame,PREPROCESS_DIMS)

		graph.queue_inference_with_fifo_elem(input_fifo, output_fifo, input_tensor, None)

		output, user_obj = output_fifo.read_elem()

		predictions = process_data(frame,output)

		# loop over our predictions
		for (i, pred) in enumerate(predictions):
			# extract prediction data for readability
			(pred_class, pred_conf, pred_boxpts) = pred

			# filter out weak detections by ensuring the `confidence`
			# is greater than the minimum confidence
			if pred_conf > 0.5:
				# print prediction to terminal
				print("[INFO] Prediction #{}: class={}, confidence={}, "
					"boxpoints={}".format(i, CLASSES[pred_class], pred_conf,
					pred_boxpts))

				# check if we should show the prediction data
				# on the frame
				# build a label consisting of the predicted class and
				# associated probability
				label = "{}: {:.2f}%".format(CLASSES[pred_class],
					pred_conf * 100)

				# extract information from the prediction boxpoints
				(ptA, ptB) = (pred_boxpts[0], pred_boxpts[1])
				ptA = (ptA[0] * DISP_MULTIPLIER, ptA[1] * DISP_MULTIPLIER)
				ptB = (ptB[0] * DISP_MULTIPLIER, ptB[1] * DISP_MULTIPLIER)
				(startX, startY) = (ptA[0], ptA[1])
				y = startY - 15 if startY - 15 > 15 else startY + 15
				
				# display the rectangle and label text
				cv2.rectangle(image_for_result, ptA, ptB,
					COLORS[pred_class], 2)
				cv2.putText(image_for_result, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[pred_class], 3)
					
		cv2.imshow("Output", image_for_result)
		

		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break

	except Exception as e:
		raise
	else:
		pass
	finally:
		pass

print("ola")
input_fifo.destroy()
output_fifo.destroy()
graph.destroy()
device.close()
device.destroy()