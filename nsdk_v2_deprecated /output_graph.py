from mvnc import mvncapi as mvnc

import numpy as np
import cv2

age_list=['(0, 2)','(4, 6)','(8, 12)','(15, 20)','(25, 32)','(38, 43)','(48, 53)','(60, 100)']
gender_list=['Male','Female']

ilsvrc_mean = np.load('graphs/age_gender_mean.npy').mean(1).mean(1) #loading the mean file

def pre_process_img_2(img,PREPROCESS_DIMS,ilsvrc_mean):
    img = cv2.resize(img, PREPROCESS_DIMS)
    img = img.astype(np.float32)
    img[:,:,0] = (img[:,:,0] - ilsvrc_mean[0])
    img[:,:,1] = (img[:,:,1] - ilsvrc_mean[1])
    img[:,:,2] = (img[:,:,2] - ilsvrc_mean[2])
    return img
    

def pre_process_img(img,PREPROCESS_DIMS):
    img = cv2.resize(img, PREPROCESS_DIMS)
    img = img - 127.5
    img = img / 127.5
    return img.astype(np.float32)

#path to the graph file
GRAPH_FILEPATH1 = 'graphs/gender_model'

device_list = mvnc.enumerate_devices()
device = mvnc.Device(device_list[0])
device.open()

with open(GRAPH_FILEPATH1, mode='rb') as f: graph_buffer1 = f.read()
graph1 = mvnc.Graph('graph1')

input_fifo, output_fifo = graph1.allocate_with_fifos(device, graph_buffer1)

cap = cv2.imread("images/tiago.jpg")
frame = cap
#frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#ret, frame = cap.read()

PREPROCESS_DIMS = (227, 227)

img = pre_process_img_2(frame,PREPROCESS_DIMS,ilsvrc_mean)

graph1.queue_inference_with_fifo_elem(input_fifo, output_fifo, img, None)
output, user_obj = output_fifo.read_elem()

print('\n------- predictions --------')
order = output.argsort()
last = len(order)-1
predicted=int(order[last])
print('the age range is ' + age_list[predicted] + ' with confidence of %3.1f%%' % (100.0*output[predicted]))
print(output)
print(output.argmax())
print(age_list[output.argmax()] )

input_fifo.destroy()
output_fifo.destroy()
graph1.destroy()
device.close()
device.destroy()

