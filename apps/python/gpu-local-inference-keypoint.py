##########################################################
####################### Libraries ########################
##########################################################
import matplotlib.pyplot as plt
from ultralytics import YOLO
import cv2
import os
import subprocess
from torchvision import transforms
import numpy as np
import time

##########################################################
###################### Parameters ########################
##########################################################

model_path="yolo-Weights/yolov8n-pose.pt"
input_video="C:/Users/kanli/Documents/GitHub/Modular-AI-Playground/mediamtx-module/sample-inputs/videos/MOT20-02-raw.webm"
output_video="C:/Users/kanli/Documents/GitHub/Modular-AI-Playground/data/inferenced_output/" + os.path.basename(model_path).split(".")[0] + os.path.basename(input_video).split(".")[0] + ".mp4"
fps = 1


##########################################################
####################### Functions ########################
##########################################################

def inference_draw(input_batch, video_writer):
    batch_results = model.track(input_batch, persist=True,conf=0.3, iou=0.5)
    # coordinates
    for index, image_result in enumerate(batch_results):
        operating_img = batch[index]
        operating_img = image_result.plot()
        video_writer.write(operating_img)

##########################################################
######################### Main ###########################
##########################################################
if __name__ == "__main__":
    # Initialize Model
    model = YOLO(model_path)
    
    # RUN
    capture = cv2.VideoCapture(input_video)
    
    # Output Size
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter(output_video, fourcc, 30, size)
    
    # Iterate
    frame_count = 0
    
    # Batch Reading
    batch = []
    
    while True:
        success, img = capture.read()
        frame_count += 1
        
        # Sanity Check
        if not success:
            print(f"Failed to grab, wrapping video at {frame_count // 30} Seconds...")
            if fps != 1:
                inference_draw(batch, output_video)
            break
        # Batching
        batch.append(img)
        if len(batch) == fps:
            print(f"Watching {frame_count // 30} Seconds...")
            inference_draw(batch, output_video)
            # Reset Batch
            batch = []
        
    capture.release()
    output_video.release()
    