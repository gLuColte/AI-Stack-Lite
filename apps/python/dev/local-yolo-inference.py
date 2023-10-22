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
from tqdm import tqdm
import pandas as pd
##########################################################
###################### Parameters ########################
##########################################################

model_path="yolov8x.pt"
input_video="/home/garylu/Github/AI-Stack-Lite/mediamtx/sample-inputs/videos/TrafficCountraw.mp4"
output_video="/home/garylu/Github/AI-Stack-Lite/" + os.path.basename(model_path).split(".")[0] + os.path.basename(input_video).split(".")[0] + ".mp4"

##########################################################
####################### Functions ########################
##########################################################

# Auxiliary functions
def risize_frame(frame, scale_percent):
    """Function to resize an image in a percent scale"""
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return resized

##########################################################
######################### Main ###########################
##########################################################
if __name__ == "__main__":
    # Initialize Model
    model = YOLO(model_path)

    # Detect Classes
    dict_classes = model.model.names
        
    ### Configurations
    #Verbose during prediction
    verbose = False
    # Scaling percentage of original frame
    scale_percent = 50

    # Reading video with cv2
    video = cv2.VideoCapture(input_video)

