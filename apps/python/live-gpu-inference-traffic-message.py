##########################################################
####################### Libraries ########################
##########################################################

import os
import cv2
import time
import copy
import uuid
import base64
import threading
import subprocess
import numpy as np
import pandas as pd
from ultralytics import YOLO
import matplotlib.pyplot as plt
from torchvision import transforms
from collections import deque
from datetime import datetime, timezone

# QUEUE
import queue

# Influx
import influxdb_client
from influxdb_client.client.write_api import SYNCHRONOUS

##########################################################
###################### Parameters ########################
##########################################################

# CAMERA LOCATION
CAMERA_LOCATION = os.environ['CAMERA_LOCATION']

# MODEL
model_path=os.environ['MODEL_PATH']

# INPUT
input_rtsp_path=os.environ['RTSP_INPUT']

# OUTPUT
output_rtsp_path=os.environ['RTSP_OUTPUT']

# Frame Queue
frame_queue = queue.Queue(maxsize=500) # !: 500 is Arbitary

# Message Queue
message_queue = queue.Queue(maxsize=100)

# Influx DB
INFLUX_DB_URL = os.environ['INFLUX_DB_URL']
INFLUX_DB_USERNAME = os.environ['INFLUX_DB_USERNAME']
INFLUX_DB_PASSWORD = os.environ['INFLUX_DB_PASSWORD']
INFLUX_DB_ORG = os.environ['INFLUX_DB_ORG']

# Generator Commands
MEASUREMENT_NAME = os.environ['MEASUREMENT_NAME']
BUCKET_NAME = os.environ['BUCKET_NAME']

# INTEREST_LINE_Y - Unscaled Raw Pixel Value
INTEREST_LINE_Y = 1500 

# SPLIT TRAFFIC LINE X - Unscaled RAW Pixel Value
TRAFFIC_LINE_X = 2000

# DEFAULT PARAMETERS
DEFAULT_MAX_INPUT_FRAME_QUEUE_SECONDS = 10
DEFAULT_SCALE_PERCENT = 50 # !: 50 Percentage means reducing
DEFAULT_CONFIDENCE = 0.7
DEFAULT_GPU_DEVICE = 0
DEFAULT_INFERENCE_VERBOSE = False

# VISUALIZATION - INTEREST
DEFAULT_INTEREST_COLOR_RGB = (36,0,199)
DEFAULT_INTEREST_LINE_SIZE = 8

# VISUALIZATION 
DEFAULT_COLOR_RGB = (199,0,57)
DEAFULT_LINE_SIZE = 5
DEFAULT_CIRCLE_RADIUS = 8
DEFAULT_CIRCLE_THICKNESS = -1
DEFAULT_TEXT_SIZE = 2
DEFAULT_FONT_SCALE = 1

##########################################################
####################### Functions ########################
##########################################################

# Auxiliary functions
def resize_frame(frame, scale_percent):
    """Function to resize an image in a percent scale"""
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return resized

def open_ffmpeg_stream_process(stream_path):
    args = ['ffmpeg',
           '-re',
           '-f', 'rawvideo',  # Apply raw video as input - it's more efficient than encoding each frame to PNG
           '-s', f'{width}x{height}', #!: Global Variables
           '-pixel_format', 'bgr24',
           '-r', f'{fps}', #!: Global Variables
           '-i', 'pipe:0',
           '-pix_fmt', 'yuv420p',
        #    '-c:v', 'h264',
           '-bufsize', '64M',
           '-maxrate', '4M',
           '-rtsp_transport', 'tcp',
           '-f', 'rtsp',
           #'-muxdelay', '0.1',
           stream_path]
    return subprocess.Popen(args, stdin=subprocess.PIPE)


##########################################################
################### Thread Functions #####################
##########################################################

def receive_function():
    
    print("Start Camera_receive Thread")
    
    cap = cv2.VideoCapture(input_rtsp_path)
    ret, frame = cap.read()
    frame_queue.put(frame)
    while ret:
        ret, frame = cap.read()
        frame_queue.put(frame)
        
def stream_function():     
    
    print("Start Stream Thread")
    
    ###############################################
    ############### Algorithm Setup ###############
    ###############################################

    # Detect Classes Names
    classes_names = model.model.names
    
    # Detect Classes IDs
    classes_IDs = [2, 3, 5, 7] 
    
    # Area of Interests
    # Y - Represents Up and down
    interest_line_y = int(INTEREST_LINE_Y * DEFAULT_SCALE_PERCENT/100) # !: Scaled based on Frame
    # X - Represents the Lane on Left and Right 
    traffic_line_x = int(TRAFFIC_LINE_X * DEFAULT_SCALE_PERCENT/100) # !: Scaled based on Frame
    
    # Offset - Gives a "THICKEN" Line offset 
    offset = int(5 * DEFAULT_SCALE_PERCENT/100 )
    
    # TOTAL Traffic Counter
    counter_in = 0
    counter_out = 0
    
    # CLASS Traffic Counter
    counter_in_classes = dict.fromkeys(classes_IDs, 0)
    counter_out_classes = dict.fromkeys(classes_IDs, 0)
    
    while True:
        if frame_queue.empty() !=True:
            # Get Frame
            frame=frame_queue.get()
            
            # Resizing frame
            operating_frame = resize_frame(frame, DEFAULT_SCALE_PERCENT)
            
            # Getting Predictions
            y_hat = model.predict(
                operating_frame, 
                conf = DEFAULT_CONFIDENCE,
                classes = classes_IDs,
                device = DEFAULT_GPU_DEVICE,
                verbose = DEFAULT_INFERENCE_VERBOSE
            )
            
            # Getting the bounding boxes, confidence and classes of the recognize objects in the current frame.
            boxes   = y_hat[0].boxes.xyxy.cpu().numpy()
            conf    = y_hat[0].boxes.conf.cpu().numpy()
            classes = y_hat[0].boxes.cls.cpu().numpy() 
            
            # Storing the above information in a dataframe
            object_dataframe = pd.DataFrame({
                'xmin': boxes[:,0], 
                'ymin': boxes[:,1], 
                'xmax': boxes[:,2], 
                'ymax': boxes[:,3], 
                'conf': conf, 
                'class': classes,
                'label': [classes_names[_] for _ in classes]
            })
            
            # Convert to INT for Coordinates
            object_dataframe[['xmin', 'ymin', 'xmax', 'ymax']] = object_dataframe[['xmin', 'ymin', 'xmax', 'ymax']].astype(int)
            
            # Drawing transition line for in\out vehicles counting 
            cv2.line(
                operating_frame, 
                (0, interest_line_y), # NOTE: this is Point, we want to draw a line, hence X = 0
                (width, interest_line_y), # NOTE: this is Point, we want to draw a line, hence X = W/E size you scaled
                DEFAULT_INTEREST_COLOR_RGB,
                DEFAULT_TEXT_SIZE
            )
            
            ###############################################
            ############ Per Object Operation #############
            ###############################################
            for index, row in enumerate(object_dataframe.iterrows()):
                # Getting the coordinates of each vehicle (row)
                xmin, ymin, xmax, ymax, confidence, class_ID, class_name = row[1]
                
                # Calculating the center of the bounding-box
                center_x, center_y = int(((xmax+xmin))/2), int((ymax+ ymin)/2)
                
                # Draw Bounding Box
                cv2.rectangle(
                    operating_frame, 
                    (xmin, ymin), 
                    (xmax, ymax), 
                    DEFAULT_COLOR_RGB, 
                    DEAFULT_LINE_SIZE
                )
                
                # Draw Centre
                cv2.circle(
                    operating_frame, 
                    (center_x,center_y), 
                    DEFAULT_CIRCLE_RADIUS,
                    DEFAULT_COLOR_RGB,
                    DEFAULT_CIRCLE_THICKNESS
                )
                
                # Write above bounding Box the name of class and Conf
                cv2.putText(
                    img=operating_frame, 
                    text=class_name+' - '+ str(round(confidence, 2)),
                    org= (xmin,ymin-10), 
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                    fontScale=DEFAULT_FONT_SCALE, 
                    color=DEFAULT_COLOR_RGB,
                    thickness=DEFAULT_TEXT_SIZE
                )
                
                # Checking if the center of recognized vehicle is in the area given by the (transition line + offset) and (transition line - offset )
                if (center_y < (interest_line_y + offset)) and (center_y > (interest_line_y - offset)):
                    if  (center_x >= 0) and (center_x <= traffic_line_x):
                        counter_in +=1
                        counter_in_classes[class_ID] += 1
                    else:
                        counter_out += 1
                        counter_out_classes[class_ID] += 1
                        
                    # Message Sending
                    message_queue.put(
                        {
                            "uuid": uuid.uuid4(),
                            "label": class_name,
                            "confidence": round(confidence, 2),
                            "captured_timestamp": datetime.now(timezone.utc).isoformat(),
                            "object_snippet": base64.b64encode(cv2.imencode('.jpg', frame[ymin:ymax, xmin:xmax])[1])
                        }
                    )
                        
                ###############################################
                ###############################################
                ###############################################

            # Write the number of vehicles in\out
            cv2.putText(
                img=operating_frame, 
                text='N. Vehicles In', 
                org= (30,30), # Coordinate of Text x,y
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                fontScale=DEFAULT_FONT_SCALE, 
                color=DEFAULT_COLOR_RGB,
                thickness=DEFAULT_TEXT_SIZE
            )
            
            cv2.putText(
                img=operating_frame, 
                text='N. Vehicles Out', 
                org= (1400, 30), # Coordinate of Text x,y
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                fontScale=DEFAULT_FONT_SCALE, 
                color=DEFAULT_COLOR_RGB,
                thickness=DEFAULT_TEXT_SIZE
            )

            # Writing the counting of type of vehicles in the corners of frame 
            xt = 40
            for _ in classes_IDs:
                xt +=30
                # IN
                cv2.putText(
                    img=operating_frame, 
                    text= f"{classes_names[_]} : {counter_in_classes[_]}", 
                    org= (30,xt), 
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                    fontScale=DEFAULT_FONT_SCALE, 
                    color=DEFAULT_COLOR_RGB,
                    thickness=DEFAULT_TEXT_SIZE
                )
                
                # OUT
                cv2.putText(
                    img=operating_frame, 
                    text= f"{classes_names[_]} : {counter_out_classes[_]}", 
                    org= (1400,xt), 
                    fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=DEFAULT_FONT_SCALE, 
                    color=DEFAULT_COLOR_RGB,
                    thickness=DEFAULT_TEXT_SIZE
                )
            
            # Writing the number of vehicles in\out
            # IN
            cv2.putText(
                img=operating_frame, 
                text=f'In:{counter_in}', 
                org= (910,interest_line_y+60),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                fontScale=DEFAULT_FONT_SCALE*2, 
                color=DEFAULT_INTEREST_COLOR_RGB,
                thickness=DEFAULT_TEXT_SIZE        )
            # OUT
            cv2.putText(
                img=operating_frame, 
                text=f'Out:{counter_out}', 
                org= (900,interest_line_y-40),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                fontScale=DEFAULT_FONT_SCALE*2, 
                color=DEFAULT_INTEREST_COLOR_RGB,
                thickness=DEFAULT_TEXT_SIZE
            )
            
            # Publish To RTSP
            ffmpeg_process.stdin.write(
                operating_frame.astype(np.uint8).tobytes()
            )

def message_function():
    """
    # Use default dictionary structure
                dict_structure = {
                    "measurement": "h2o_feet",
                    "tags": {"location": "coyote_creek"},
                    "fields": {"water_level": 1.0},
                    "time": 1
                }
                point = Point.from_dict(dict_structure, WritePrecision.NS)
    """
    
    print("Start Message Thread")
    
    # Initialize Influx DB
    client = influxdb_client.InfluxDBClient(
        url=INFLUX_DB_URL,
        username=INFLUX_DB_USERNAME, 
        password=INFLUX_DB_PASSWORD,
        org=INFLUX_DB_ORG
    )
    write_api = client.write_api(write_options=SYNCHRONOUS)
    
    message_counter = 1
    while True:
        if message_queue.empty() !=True:
            # Get Frame
            operating_message = message_queue.get()
            
            # Check
            print(f"Retrieved Message:\n{operating_message}")
            
            # Creating Data Point
            data_dict = {
                "measurement": MEASUREMENT_NAME,
                "tags": {
                    "location": CAMERA_LOCATION
                },
                "fields": {k:v for k,v in operating_message.items()}
            }
            
            # point
            point = influxdb_client.Point(data_dict)
            
            # Write
            write_api.write(bucket=BUCKET_NAME, org=INFLUX_DB_ORG, record=data_point)
            print(f"Inserted {count} records")
            message_counter+=1
    

##########################################################
######################### Main ###########################
##########################################################
if __name__ == "__main__":
    
    ###############################################
    ############## Input Video Setup ##############
    ###############################################
    
    # Reading video with cv2
    input_video = cv2.VideoCapture(input_rtsp_path, cv2.CAP_FFMPEG)
    
    # Video Dimensions
    height = int(input_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(input_video.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = input_video.get(cv2.CAP_PROP_FPS)
    
    # Check
    print(f'[INFO] - Original Dim: {(width, height)}, FPS: {fps}' )
    
    # Scaling Video for better performance 
    if DEFAULT_SCALE_PERCENT != 100:
        width = int(width * DEFAULT_SCALE_PERCENT / 100)
        height = int(height * DEFAULT_SCALE_PERCENT / 100)
        print('[INFO] - Dim Scaled: ', (width, height))
        
    ###############################################
    ############ Output Stream Setup ##############
    ###############################################
    ffmpeg_process = open_ffmpeg_stream_process(output_rtsp_path)
    
    ###############################################
    ############# Algorithm Execution #############
    ###############################################
    
    # Initialize Model - Trigger Download Before threads
    model = YOLO(model_path)

    # Defining Threads
    receive_thread = threading.Thread(target=receive_function)
    stream_thread = threading.Thread(target=stream_function)

    # Starting Threads
    receive_thread.start()
    stream_thread.start()