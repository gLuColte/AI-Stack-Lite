##########################################################
####################### Libraries ########################
##########################################################

import os
import cv2
import time

import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# YOLO
from ultralytics import YOLO

# Importing
from inference_functions import *

# Multi Processing
import multiprocessing

##########################################################
###################### Parameters ########################
##########################################################
    
# MODEL
DEFAULT_MODEL_PATH = os.environ['MODEL_PATH']
CLASS_IDS = [int(_) for _ in os.environ['CLASS_IDS'].split(",")]

# INPUT
INPUT_RTSP_URL = os.environ['RTSP_INPUT']

# OUTPUT
OUTPUT_RTSP_URL = os.environ['RTSP_OUTPUT']

# INTEREST_LINE_Y - Unscaled Raw Pixel Value
INTEREST_LINE_COORDINATES = tuple([int(_) for _ in os.environ['INTEREST_LINE_COORDINATES'].split(",")]) # ! One has to be zero

# SPLIT TRAFFIC LINE X - Unscaled RAW Pixel Value
TRAFFIC_LINE_COORDINATES = tuple([int(_) for _ in os.environ['TRAFFIC_LINE_COORDINATES'].split(",")])

# DEFAULT PARAMETERS
DEFAULT_SCALE_PERCENT = int(os.environ['SCALE_PERCENT']) # !: 50 Percentage means reducing
DEFAULT_CONFIDENCE_THRESHOLD = 0.7
DEFAULT_GPU_DEVICE = 0 if 'DEFAULT_GPU_DEVICE' not in os.environ.keys() else int(os.environ["DEFAULT_GPU_DEVICE"])
DEFAULT_INFERENCE_VERBOSE = False

# VISUALIZATION - INTEREST
DEFAULT_INTEREST_COLOR_RGB = (36,0,199)
DEFAULT_INTEREST_LINE_SIZE = 8

# VISUALIZATION 
DEFAULT_COLOR_RGB = (199,0,57)
DEAFULT_LINE_SIZE = 5 if 'DEFAULT_LINE_SIZE' not in os.environ.keys() else int(os.environ["DEFAULT_LINE_SIZE"])
DEFAULT_CIRCLE_RADIUS = 8
DEFAULT_CIRCLE_THICKNESS = -1
DEFAULT_TEXT_SIZE = 2 if 'DEFAULT_TEXT_SIZE' not in os.environ.keys() else int(os.environ["DEFAULT_TEXT_SIZE"])
DEFAULT_FONT_SCALE = 1 if 'DEFAULT_FONT_SCALE' not in os.environ.keys() else int(os.environ["DEFAULT_FONT_SCALE"])

# DEFAULT OFFSET
DEFAULT_OFFSET = 8 if 'DEFAULT_OFFSET' not in os.environ.keys() else int(os.environ["DEFAULT_OFFSET"])

##########################################################
######################### Main ###########################
##########################################################
if __name__ == "__main__":
    
    ###############################################
    ############## Input Video Setup ##############
    ###############################################
    
    # Reading video with cv2 for initial setup
    input_video_cap = cv2.VideoCapture(INPUT_RTSP_URL, cv2.CAP_FFMPEG)
    
    # Video Dimensions
    LIVE_STREAM_HEIGHT = int(input_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    LIVE_STREAM_WIDTH = int(input_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    LIVE_STREAM_FPS = input_video_cap.get(cv2.CAP_PROP_FPS)
    
    # Check
    print(f'[INFO] - Original Dim (W x H): {(LIVE_STREAM_WIDTH, LIVE_STREAM_HEIGHT)}, FPS: {LIVE_STREAM_FPS}' )
    
    # Scaling Video for better performance 
    if DEFAULT_SCALE_PERCENT != 100:
        LIVE_STREAM_HEIGHT = int(LIVE_STREAM_HEIGHT * DEFAULT_SCALE_PERCENT / 100)
        LIVE_STREAM_WIDTH = int(LIVE_STREAM_WIDTH * DEFAULT_SCALE_PERCENT / 100)
        print('[INFO] - Dim Scaled (W x H): ', (LIVE_STREAM_WIDTH, LIVE_STREAM_HEIGHT))
    
    # Releasing Cap
    input_video_cap.release()        
    
    ###############################################
    ############## Wrapping Info JSON #############
    ###############################################
    
    # Post Processing Args input
    post_proccessing_input_dict = {
        "model_path": DEFAULT_MODEL_PATH,
        "height": LIVE_STREAM_HEIGHT,
        "width": LIVE_STREAM_WIDTH,
        "fps": LIVE_STREAM_FPS,
        "output_rtsp_url": OUTPUT_RTSP_URL,
        "scale_percent": DEFAULT_SCALE_PERCENT,
        "confidence_threshold": DEFAULT_CONFIDENCE_THRESHOLD,
        "gpu_device": DEFAULT_GPU_DEVICE,
        
        # Class IDs
        "class_IDs": CLASS_IDS,
        
        # Drawings
        "interest_color_rgb": DEFAULT_INTEREST_COLOR_RGB,
        "interest_line_size": DEFAULT_INTEREST_LINE_SIZE,
        
        # VISUALIZATION 
        "color_rgb": DEFAULT_COLOR_RGB,
        "line_size": DEAFULT_LINE_SIZE,
        "circle_radius": DEFAULT_CIRCLE_RADIUS,
        "circle_thickness": DEFAULT_CIRCLE_THICKNESS,
        "text_size": DEFAULT_TEXT_SIZE,
        "font_scale": DEFAULT_FONT_SCALE,
        
        # LINEs
        "interest_line_coordinates": INTEREST_LINE_COORDINATES,
        "traffic_line_coordinates": TRAFFIC_LINE_COORDINATES,
        "offset": DEFAULT_OFFSET,
    }
    
    # Message Input
    # message_input_dict = {
        
    # }
    
    ###############################################
    ############# Algorithm Execution #############
    ###############################################
    
    # Frame Queue
    frame_queue = multiprocessing.Queue(maxsize=60) 
    message_queue = multiprocessing.Queue(maxsize=60)

    # Defining Threads
    frame_receive_process = multiprocessing.Process(
        target=frame_receive_function, 
        args=(frame_queue, INPUT_RTSP_URL)
    )
    post_processing_process = multiprocessing.Process(
        target=post_processing_function, 
        args=(frame_queue, message_queue, post_proccessing_input_dict)
    )
    # message_process = multiprocessing.Process(
    #     target=message_function,
    #     args=(message_queue, message_input_dict)
    # )
    
    
    # Starting
    frame_receive_process.start()
    post_processing_process.start()
    # message_process.start()