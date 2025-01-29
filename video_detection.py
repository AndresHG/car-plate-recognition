import os
from collections import defaultdict
import xml.etree.ElementTree as xet
import pytesseract as pt

import cv2
import numpy as np
import pandas as pd

import plotly.express as px
import matplotlib.pyplot as plt

from glob import glob
from skimage import io
from shutil import copy
import time

import torch

torch.set_num_threads(2) 

def get_detections(img, net):
    # 1.CONVERT IMAGE TO YOLO FORMAT
    image = img.copy()
    row, col, d = image.shape

    max_rc = max(row, col)
    input_image = np.zeros((max_rc, max_rc, 3),dtype=np.uint8)
    input_image[:row, :col] = image

    # 2. GET PREDICTION FROM YOLO MODEL
    # pred_image = cv2.resize(input_image, dsize=(INPUT_WIDTH,INPUT_HEIGHT), interpolation=cv2.INTER_CUBIC)
    preds = net(input_image)
    detections = preds.pandas().xyxy[0]
    
    return input_image, detections

def non_maximum_supression(input_image, detections):
    # Filter predictions/detections based on threshold
    boxes = []
    confidences = []

    for i, row in detections.iterrows():
        confidence = row.confidence # confidence of detecting license plate
        if confidence > 0.4:
            x0, y0, x1, y1 = int(row.xmin), int(row.ymin), int(row.xmax), int(row.ymax)

            box = np.array([x0, y0, x1, y1])

            confidences.append(confidence)
            boxes.append(box)

    boxes_np = np.array(boxes).tolist()
    confidences_np = np.array(confidences).tolist()
    
    index = cv2.dnn.NMSBoxes(boxes_np,confidences_np,0.25,0.45)
    
    return boxes_np, confidences_np, index

def drawings(image,boxes_np,confidences_np,index, device_name, fps):
    img_w = image.shape[1]
    
    # Add device
    cv2.rectangle(image,(10,10),(500,80),(255,0,255),-1)
    cv2.putText(image, device_name, (20, 60),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=2)

    # Add time elapsed
    cv2.rectangle(image,(img_w-500,10),(img_w-20,80),(255,0,0),-1) # BGR
    cv2.putText(image, "FPS: " + str(round(fps, 2)), (img_w-490, 60),cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), thickness=2)
    
    # 5. Drawings
    for ind in index:
        
        x,y,w,h =  boxes_np[ind]
        bb_conf = confidences_np[ind]
        conf_text = 'plate: {:.0f}%'.format(bb_conf*100)
        # license_text = extract_text(image,boxes_np[ind])

        cv2.rectangle(image,(x,y),(w,h),(255,0,255),2)
        cv2.rectangle(image,(x,y-30),(w,y),(255,0,255),-1)
        # cv2.rectangle(image,(x,h),(w,h+25),(0,0,0),-1)


        cv2.putText(image,conf_text,(x, y-10),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)
        # cv2.putText(image,license_text,(x, h+27),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 1)
        
    return image

# predictions flow with return result
def yolo_predictions(img,net):
    start = time.time()
    # step-1: detections
    input_image, detections = get_detections(img, net)
    # step-2: NMS
    boxes_np, confidences_np, index = non_maximum_supression(input_image, detections)
    
    if next(net.parameters()).is_cuda:
        device_name = "GPU"
    else:
        device_name = "CPU"
    
    done = time.time()
    elapsed = done - start
    
    # step-3: Drawings
    result_img = drawings(img, boxes_np, confidences_np, index, device_name, 1/elapsed)
    return result_img

# extrating text
def extract_text(image, bbox):
    x,y,w,h = bbox
    roi = image[y:y+h, x:x+w]
    
    if 0 in roi.shape:
        return 'no number'
    
    else:
        text = pt.image_to_string(roi)
        text = text.strip()
        
        return text


# GENERAL VARS
SAVE_VIDEO = True
DEVICE="cpu"
    
# LOAD YOLO MODELLoad the best Yolo model
CKPT_PATH = './yolov5/runs/train/Model3/weights/best.pt'
net = torch.hub.load(
    './yolov5',
    'custom',
    path=CKPT_PATH,
    source='local',
)
net = net.to(DEVICE)

# Create input video reader
cap = cv2.VideoCapture('./data/TEST/TEST.mp4')
save_path = "output.mp4"

# Define parameters for output video
width = 1700.0  # float
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  - 400# float
fps = cap.get(cv2.CAP_PROP_FPS) if DEVICE == "cuda" else 7

if SAVE_VIDEO:
    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height))
    )

# Start loop to read the video and predict
while True:
    ret, frame = cap.read()
    # Define ROI
    frame = frame[400:, :1700]
    
    if ret == False:
        print('Unable to read video or video finished')
        break

    results = yolo_predictions(frame, net)

    # Write the frame into the file 'output.mp4'
    if SAVE_VIDEO:
        vid_writer.write(results)
    else:
        cv2.namedWindow('YOLO', cv2.WINDOW_KEEPRATIO)
        cv2.imshow('YOLO', results)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

cv2.destroyAllWindows()
cap.release()
