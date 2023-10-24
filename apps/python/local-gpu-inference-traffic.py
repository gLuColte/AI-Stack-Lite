##########################################################
####################### Libraries ########################
##########################################################

import os
import cv2
import time
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO
import matplotlib.pyplot as plt
from torchvision import transforms
from collections import deque

##########################################################
###################### Parameters ########################
##########################################################

# MODEL
model_path="yolov8x.pt"

# INPUT
input_video_path="/home/garylu/Github/AI-Stack-Lite/mediamtx/sample-inputs/videos/TrafficCountraw.mp4"

# OUTPUT
output_video_codec = "mp4v"
output_dir_path="/home/garylu/Github/AI-Stack-Lite/data/"
output_video_filename = os.path.basename(model_path).split(".")[0] + os.path.basename(input_video_path).split(".")[0] + ".mp4"
output_video_tmp_path=output_dir_path + 'tmp_' + output_video_filename


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

##########################################################
######################### Main ###########################
##########################################################
if __name__ == "__main__":
    
    ###############################################
    ############## Input Video Setup ##############
    ###############################################
    
    # Reading video with cv2
    input_video = cv2.VideoCapture(input_video_path)
    
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
    ############### Algorithm Setup ###############
    ###############################################
    
    # Initialize Model
    model = YOLO(model_path)

    # Detect Classes Names
    classes_names = model.model.names
    
    # Detect Classes IDs
    classes_IDs = [2, 3, 5, 7] 
    
    # Queues
    input_frame_queue = deque(maxlen=int(fps*DEFAULT_MAX_INPUT_FRAME_QUEUE_SECONDS)) # !: FPS
    output_frame_list = []
    
    # Area of Interests
    # Y - Represents Up and down
    interest_line_y = int(1500 * DEFAULT_SCALE_PERCENT/100) # !: Scaled based on Frame
    # X - Represents the Lane on Left and Right 
    interest_line_x = int(2000 * DEFAULT_SCALE_PERCENT/100) # !: Scaled based on Frame
    
    # Offset - Gives a "THICKEN" Line offset 
    offset = int(8 * DEFAULT_SCALE_PERCENT/100 )
    
    # TOTAL Traffic Counter
    counter_in = 0
    counter_out = 0
    
    # CLASS Traffic Counter
    counter_in_classes = dict.fromkeys(classes_IDs, 0)
    counter_out_classes = dict.fromkeys(classes_IDs, 0)
    
    ###############################################
    ############# Output Video Setup ##############
    ###############################################
    
    # Output Video Writer
    output_video = cv2.VideoWriter(
        output_video_tmp_path, 
        cv2.VideoWriter_fourcc(*output_video_codec), 
        fps, (width, height)
    )
    
    ###############################################
    ############# Algorithm Execution #############
    ###############################################
    
    # Iterate based on Total Frame Count
    for frame_count in tqdm(range(int(input_video.get(cv2.CAP_PROP_FRAME_COUNT)))):
        
        # Reading frame from Video
        read_indicator, frame = input_video.read()
        
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
            (int(4500 * DEFAULT_SCALE_PERCENT/100 ), interest_line_y), # NOTE: this is Point, we want to draw a line, hence X = W/E size you scaled
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
                if  (center_x >= 0) and (center_x <= interest_line_x):
                    counter_in +=1
                    counter_in_classes[class_ID] += 1
                else:
                    counter_out += 1
                    counter_out_classes[class_ID] += 1
                    
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
            thickness=1
        )
        
        cv2.putText(
            img=operating_frame, 
            text='N. Vehicles Out', 
            org= (int(2800 * DEFAULT_SCALE_PERCENT/100 ), 30), # Coordinate of Text x,y
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
                text= f"{_} : {counter_in_classes[_]}", 
                org= (30,xt), 
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                fontScale=DEFAULT_FONT_SCALE, 
                color=DEFAULT_COLOR_RGB,
                thickness=DEFAULT_TEXT_SIZE
            )
            
            # OUT
            cv2.putText(
                img=operating_frame, 
                text= f"{_} : {counter_out_classes[_]}", 
                org= (int(2800 * DEFAULT_SCALE_PERCENT/100 ),xt), 
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
            org= (int(1820 * DEFAULT_SCALE_PERCENT/100 ),interest_line_y+60),
            fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
            fontScale=DEFAULT_FONT_SCALE*2, 
            color=DEFAULT_INTEREST_COLOR_RGB,
            thickness=DEFAULT_TEXT_SIZE        )
        # OUT
        cv2.putText(
            img=operating_frame, 
            text=f'Out:{counter_out}', 
            org= (int(1800 * DEFAULT_SCALE_PERCENT/100 ),interest_line_y-40),
            fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
            fontScale=DEFAULT_FONT_SCALE*2, 
            color=DEFAULT_INTEREST_COLOR_RGB,
            thickness=DEFAULT_TEXT_SIZE
        )
        
        #Saving frames in a list 
        output_frame_list.append(operating_frame)
        #saving transformed frames in a output video formaat
        output_video.write(operating_frame)
        
    # Releasing the video    
    output_video.release()
    
    # Fixing Codec Issue
    output_video_converted_path=output_dir_path + output_video_filename
    subprocess.run(
        ["ffmpeg",  "-i", output_video_tmp_path,"-crf","18","-preset","veryfast","-hide_banner","-loglevel","error","-vcodec","libx264",output_video_converted_path]
    )
    os.remove(output_video_tmp_path)
    