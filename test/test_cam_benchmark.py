#import libs
from mvnc import mvncapi as mvnc
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import numpy as np
import time
import cv2
import os
#mode of operating -> "image" for test images/ "cam" for video 
mode = "image"
#mode = "cam"

#directory with images
directory = ("images")

#import labels for prediction boxes 
CLASSES = ("background", "aeroplane", "bicycle", "bird",
	"boat", "bottle", "bus", "car", "cat", "chair", "cow",
	"diningtable", "dog", "horse", "motorbike", "person",
	"pottedplant", "sheep", "sofa", "train", "tvmonitor")
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# frame dimensions should be sqaure
PREPROCESS_DIMS = (300, 300)
DISPLAY_DIMS = (600, 600)

# calculate the multiplier needed to scale the bounding boxes
DISP_MULTIPLIER = DISPLAY_DIMS[0] // PREPROCESS_DIMS[0]

def preprocess_image(input_image):
	# preprocess the image
	preprocessed = cv2.resize(input_image, PREPROCESS_DIMS)
	preprocessed = preprocessed - 127.5
	preprocessed = preprocessed * 0.007843
	preprocessed = preprocessed.astype(np.float16)

	# return the image to the calling function
	return preprocessed



def predict(image, graph):
	# preprocess the image
	start = time.clock()
	image = preprocess_image(image)
	time_consumed_proc = (time.clock() - start)

	# send the image to the NCS and run a forward pass to grab the
	# network predictions
	start = time.clock()
	graph.LoadTensor(image, None)
	(output, _) = graph.GetResult()
	time_consumed_load_and_get = (time.clock() - start)

	# grab the number of valid object predictions from the output,
	# then initialize the list of predictions
	num_valid_boxes = output[0]
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
	return predictions, time_consumed_proc, time_consumed_load_and_get

# grab a list of all NCS devices plugged in to USB
print("[INFO] finding NCS devices...")
devices = mvnc.EnumerateDevices()

# if no devices found, exit the script
if len(devices) == 0:
	print("[INFO] No devices found. Please plug in a NCS")
	quit()

	# use the first device since this is a simple test script
# (you'll want to modify this is using multiple NCS devices)
print("[INFO] found {} devices. device0 will be used. "
	"opening device0...".format(len(devices)))
device = mvnc.Device(devices[0])
device.OpenDevice()

# open the CNN graph file
print("[INFO] loading the graph file into RPi memory...")
with open("graphs/mobilenetgraph", mode="rb") as f:
	graph_in_memory = f.read()

# load the graph into the NCS
print("[INFO] allocating the graph on the NCS...")
graph = device.AllocateGraph(graph_in_memory)

print("[INFO] starting the video stream and FPS counter...")

#load an image to test -> to-do: make to pass an image as argument to program


time.sleep(1)
fps = FPS().start()
fimt = []

if mode == "cam":
	cap = cv2.VideoCapture(0)
	while True:
		try:
			#for video stream
			ret, frame = cap.read()
			image_for_result = frame.copy()
			image_for_result = cv2.resize(image_for_result, DISPLAY_DIMS)

			start = time.clock()
			# use the NCS to acquire predictions
			predictions, time_consumed_proc = predict(frame, graph)

			time_consumed_pred= (time.clock() - start)

			start = time.clock()
			# loop over our predictions
			for (i, pred) in enumerate(predictions):
				# extract prediction data for readability
				(pred_class, pred_conf, pred_boxpts) = pred

				# filter out weak detections by ensuring the `confidence`
				# is greater than the minimum confidence
				if pred_conf > 0.5:
					# print prediction to terminal
					
					#print("[INFO] Prediction #{}: class={}, confidence={}, "
					#	"boxpoints={}".format(i, CLASSES[pred_class], pred_conf,
					#	pred_boxpts))

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

			time_consumed_draw= (time.clock() - start)

			print("Image: {} prediction: {} drawing: {}".format(filename,time_consumed_pred,time_consumed_draw))

			key = cv2.waitKey(1) & 0xFF
			if key == ord("q"):

					break

			cv2.imshow("Output", image_for_result)
			#cv2.imwrite("cam/{}_{}.jpg".format(filename,"saved"),image_for_result)	

			fps.update()

		except Exception as e:
			raise
		else:
			pass
		finally:
			pass


if mode == "image":
	for filename in os.listdir(directory):
		cap = cv2.imread(os.path.join(directory, filename))	

		frame = cap
		image_for_result = frame.copy()
		image_for_result = cv2.resize(image_for_result, DISPLAY_DIMS)

		start = time.clock()
		# use the NCS to acquire predictions
		predictions, time_consumed_proc, time_consumed_load_and_get = predict(frame, graph)

		time_consumed_pred= (time.clock() - start)
		

		start = time.clock()
		# loop over our predictions
		for (i, pred) in enumerate(predictions):
			# extract prediction data for readability
			(pred_class, pred_conf, pred_boxpts) = pred

			# filter out weak detections by ensuring the `confidence`
			# is greater than the minimum confidence
			if pred_conf > 0.5:
				# print prediction to terminal
				
				#print("[INFO] Prediction #{}: class={}, confidence={}, "
				#	"boxpoints={}".format(i, CLASSES[pred_class], pred_conf,
				#	pred_boxpts))

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

		time_consumed_draw= (time.clock() - start)
		
		print("Image: {} prediction: {} drawing: {} proc: {} load_get: {}".format(filename,time_consumed_pred,time_consumed_draw, time_consumed_proc, time_consumed_load_and_get))

		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):

				break

		#cv2.imshow("Output", image_for_result)
		#cv2.imwrite("images/{}_{}.jpg".format(filename,"saved"),image_for_result)	

		#fps.update()




