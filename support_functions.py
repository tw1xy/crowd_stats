
import numpy as np
import cv2



CLASSES = ("background", "aeroplane", "bicycle", "bird",
"boat", "bottle", "bus", "car", "cat", "chair", "cow",
"diningtable", "dog", "horse", "motorbike", "person",
"pottedplant", "sheep", "sofa", "train", "tvmonitor")

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

#loading the mean file
ilsvrc_mean = np.load('graphs/age_gender_mean.npy').mean(1).mean(1) 

def process_prediction(output,image_shape,DISP_MULT):
    num_valid_boxes = int(output[0])
    predictions = []
    boxes_only = []
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

        (h, w) = image_shape
        #(h, w) = img.shape[:2]
        x1 = max(0, int(output[base_index + 3] * w))
        y1 = max(0, int(output[base_index + 4] * h))
        x2 = min(w,	int(output[base_index + 5] * w))
        y2 = min(h,	int(output[base_index + 6] * h))
        
        pred_class = int(output[base_index + 1])
        pred_conf = output[base_index + 2]
        pred_boxpts = ((x1, y1), (x2, y2))
        prediction = (pred_class, pred_conf, pred_boxpts)
        
        
        if pred_conf > 0.6 and pred_class == 15 or pred_class == 1: #only people output
            predictions.append(prediction)
            
            boxes_only.append((int(x1*DISP_MULT[0]), int(y1*DISP_MULT[1]), 
                int(x2*DISP_MULT[0]), int(y2*DISP_MULT[1])))
        else:
            pass
        
    return predictions, boxes_only

def person_boxes_only(predictions):
    boxes_only = []
    for (i, pred) in enumerate(predictions):
        (pred_class, pred_conf, pred_boxpts) = pred
        if pred_conf > 0.7 and pred_class == 15:
            boxes_only.append(pred_boxpts[0]+pred_boxpts[1])
        else:
            pass
    return boxes_only

def draw_boxes(predictions,image_for_result,DISP_MULT):
    DISP_MULTIPLIER_X, DISP_MULTIPLIER_Y = DISP_MULT
    
    for (i, pred) in enumerate(predictions):
        (pred_class, pred_conf, pred_boxpts) = pred

        if pred_conf > 0.5:
            #print("[INFO] Prediction #{}: class={}, confidence={}, "
             #   "boxpoints={}".format(i, CLASSES[pred_class], pred_conf,pred_boxpts))

            label = "{}: {:.2f}%".format(CLASSES[pred_class],
                pred_conf * 100)

            (ptA, ptB) = (pred_boxpts[0], pred_boxpts[1])
            ptA = (int(ptA[0] * DISP_MULTIPLIER_X), int(ptA[1] * DISP_MULTIPLIER_Y))
            ptB = (int(ptB[0] * DISP_MULTIPLIER_X), int(ptB[1] * DISP_MULTIPLIER_Y))
            (startX, startY) = (ptA[0], ptA[1])
            y = startY - 15 if startY - 15 > 15 else startY + 15

            # cv2.rectangle(image_for_result, ptA, ptB,
            #     COLORS[pred_class], 2)
            # cv2.putText(image_for_result, label, (startX, y),
            #     cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[pred_class], 2)

        return image_for_result

def draw_output(predictions,image_for_result,DISP_MULT):
    DISP_MULTIPLIER_X, DISP_MULTIPLIER_Y = DISP_MULT

    for (i, pred) in enumerate(predictions):
        (pred_class, pred_conf, pred_boxpts) = pred

        if pred_conf > 0.5:
            # print("[INFO] Prediction #{}: class={}, confidence={}, "
            #     "boxpoints={}".format(i, CLASSES[pred_class], pred_conf,pred_boxpts))

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

def process_age(output):
    age_index = output.argmax()
    age = age_list[age_index]
    return age

def process_gender(output):
    gender_index = output.argmax()
    gender = gender_list[gender_index]
    return gender

def pre_process_img(img,PREPROCESS_DIMS):
    #USE WITH MOBILE_SSD (many classes, person on)
    img = cv2.resize(img, PREPROCESS_DIMS)
    img = img - 127.5
    img = img / 127.5
    return img.astype(np.float32)

def pre_process_img_2(img,PREPROCESS_DIMS,ilsvrc_mean):
    #USE WITH CAFFE_NET (age and gender)
    img = cv2.resize(img, PREPROCESS_DIMS)
    img = img.astype(np.float32)
    img[:,:,0] = (img[:,:,0] - ilsvrc_mean[0])
    img[:,:,1] = (img[:,:,1] - ilsvrc_mean[1])
    img[:,:,2] = (img[:,:,2] - ilsvrc_mean[2])
    return img
