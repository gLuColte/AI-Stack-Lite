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

##########################################################
####################### Functions ########################
##########################################################

def resize_frame(frame, scale_percent):
    """Function to resize an image in a percent scale"""
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return resized

def open_ffmpeg_stream_process(stream_path, height, width, fps):
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
################## Counter Functions #####################
##########################################################

# ! Not Ideal Method
def xx_counter_function(
    center_coordinate, 
    interest_coordainte, 
    traffic_coordinate, 
    input_class_ID,
    input_counter_in, 
    input_counter_in_classes,
    input_counter_out, 
    input_counter_out_classes,
    input_offset
    ):
    if (center_coordinate[0] < (interest_coordainte[0] + input_offset)) and (center_coordinate[0] > (interest_coordainte[0] - input_offset)):
        if  (center_coordinate[0] >= 0) and (center_coordinate[0] <= traffic_coordinate[0]):
            input_counter_in +=1
            input_counter_in_classes[input_class_ID] += 1
        else:
            input_counter_out += 1
            input_counter_out_classes[input_class_ID] += 1
    return input_counter_in, input_counter_in_classes, input_counter_out, input_counter_out_classes

def xy_counter_function(
    center_coordinate, 
    interest_coordainte, 
    traffic_coordinate, 
    input_class_ID,
    input_counter_in, 
    input_counter_in_classes,
    input_counter_out, 
    input_counter_out_classes,
    input_offset
    ):
    if (center_coordinate[0] < (interest_coordainte[0] + input_offset)) and (center_coordinate[0] > (interest_coordainte[0] - input_offset)):
        if  (center_coordinate[1] >= 0) and (center_coordinate[1] <= traffic_coordinate[1]):
            input_counter_in +=1
            input_counter_in_classes[input_class_ID] += 1
        else:
            input_counter_out += 1
            input_counter_out_classes[input_class_ID] += 1
    return input_counter_in, input_counter_in_classes, input_counter_out, input_counter_out_classes


def yy_counter_function(
    center_coordinate, 
    interest_coordainte, 
    traffic_coordinate, 
    input_class_ID,
    input_counter_in, 
    input_counter_in_classes,
    input_counter_out, 
    input_counter_out_classes,
    input_offset
    ):
    if (center_coordinate[1] < (interest_coordainte[1] + input_offset)) and (center_coordinate[1] > (interest_coordainte[1] - input_offset)):
        if  (center_coordinate[1] >= 0) and (center_coordinate[1] <= traffic_coordinate[1]):
            input_counter_in +=1
            input_counter_in_classes[input_class_ID] += 1
        else:
            input_counter_out += 1
            input_counter_out_classes[input_class_ID] += 1
    return input_counter_in, input_counter_in_classes, input_counter_out, input_counter_out_classes


def yx_counter_function(
    center_coordinate, 
    interest_coordainte, 
    traffic_coordinate, 
    input_class_ID,
    input_counter_in, 
    input_counter_in_classes,
    input_counter_out, 
    input_counter_out_classes,
    input_offset
    ):
    if (center_coordinate[1] < (interest_coordainte[1] + input_offset)) and (center_coordinate[1] > (interest_coordainte[1] - input_offset)):
        if  (center_coordinate[0] >= 0) and (center_coordinate[0] <= traffic_coordinate[0]):
            input_counter_in +=1
            input_counter_in_classes[input_class_ID] += 1
        else:
            input_counter_out += 1
            input_counter_out_classes[input_class_ID] += 1    
    return input_counter_in, input_counter_in_classes, input_counter_out, input_counter_out_classes

##########################################################
################### Thread Functions #####################
##########################################################

def frame_receive_function(input_frame_queue, input_rtsp_path):
    print(f"Started Frame Recevie Function with: {input_rtsp_path}")
    
    # Initialize
    stream_cap = cv2.VideoCapture(input_rtsp_path)
    
    # Begin Iteration:
    print("Begin Frame Receive Iteration")
    while True:
        try:
            ret, frame = stream_cap.read()
            input_frame_queue.put(frame)
        except Exception as error:
            print(f"Frame Receive Function Error:\n{error}")
            time.sleep(5)
            continue

def post_processing_function(input_frame_queue, input_message_queue, input_json):
    
    print(f"Started Post Processing Function with inputs:\n{input_json}")
    ###############################################
    ############# Reading Variables ###############
    ###############################################
    
    model_path = input_json["model_path"]
    height = input_json["height"]
    width = input_json["width"]
    fps = input_json["fps"]
    output_rtsp_url = input_json["output_rtsp_url"]
    scale_percent = input_json["scale_percent"]
    confidence_threshold = input_json["confidence_threshold"]
    gpu_device = input_json["gpu_device"]
    interest_color_rgb = input_json["interest_color_rgb"]
    interest_line_size = input_json["interest_line_size"]
    color_rgb = input_json["color_rgb"]
    line_size = input_json["line_size"]
    circle_radius = input_json["circle_radius"]
    circle_thickness = input_json["circle_thickness"]
    text_size = input_json["text_size"]
    font_scale = input_json["font_scale"]
    
    # LINEs
    interest_line_coordinates = input_json["interest_line_coordinates"]
    traffic_line_coordinates = input_json["traffic_line_coordinates"]
        
    ###############################################
    ############### Algorithm Setup ###############
    ###############################################
    
    # Initialize Model - Trigger Download Before threads
    model = YOLO(model_path)
    
    # Detect Classes Names
    classes_names = model.model.names
    
    # Detect Classes IDs
    classes_IDs = input_json["class_IDs"]
    
    # Offset - Gives a "THICKEN" Line offset 
    offset = int(input_json["offset"] * scale_percent/100 )
    
    # TOTAL Traffic Counter
    counter_in = 0
    counter_out = 0
    
    # CLASS Traffic Counter
    counter_in_classes = dict.fromkeys(classes_IDs, 0)
    counter_out_classes = dict.fromkeys(classes_IDs, 0)
    
    ###############################################
    ################# Lines Setup #################
    ###############################################
    
    # Frame Reference
    frame_size = [width, height]    
    
    # Function Indicator
    function_indicator_list = [None, None]
    
    # LINE OF INTEREST
    interest_line_coordinate_1 = [0, 0]
    interest_line_coordinate_2 = frame_size
    for index, coordinate in enumerate(interest_line_coordinates):
        if coordinate != 0:
            interest_line_coordinate_1[index] = int(coordinate * scale_percent/100)
            interest_line_coordinate_2[index] = int(coordinate * scale_percent/100)
            function_indicator_list[0] = 'x' if index == 0 else 'y'
            
            # Total Coordinate
            if index == 1:
                total_in_coordinate = tuple([60, int(coordinate * scale_percent/100) + 60])
                total_out_coordinate = tuple([60,  int(coordinate * scale_percent/100) - 60])
            else:
                total_in_coordinate = tuple([60, int(height/2)])
                total_out_coordinate = tuple([int(width/2),  int(height/2)])
            break
    # Convert to tuple
    interest_line_coordinate_1 = tuple(interest_line_coordinate_1)
    interest_line_coordinate_2 = tuple(interest_line_coordinate_2)

    
    # TRAFFIC SPLITTER
    traffic_line_coordinate_1 = [0, 0]
    for index, coordinate in enumerate(traffic_line_coordinates):
        if coordinate != 0:
            traffic_line_coordinate_1[index] = int(coordinate * scale_percent/100)
            function_indicator_list[1] = 'x' if index == 0 else 'y'
            break
    # Convert to tuple
    traffic_line_coordinate_1 = tuple(traffic_line_coordinate_1)
    
    # Summary Text Coordinates
    detail_in_coordinate = (30,30)
    detail_out_coordinate = (int(0.60*width),30)
    
    # Functions to call for counter
    FUNCTION_INDICATOR = ''.join(function_indicator_list)
    function_dictionary = {
        'xx' : xx_counter_function,
        'xy' : xy_counter_function,
        'yy' : yy_counter_function,
        'yx' : yx_counter_function    
    }
    
    # Check Coordinates:
    print(f"Interest Line Coordinates: {interest_line_coordinate_1} and {interest_line_coordinate_2}")
    print(f"Traffic Line Coordinates: {traffic_line_coordinate_1}")
    
    ###############################################
    ############ Output Stream Setup ##############
    ###############################################
    ffmpeg_process = open_ffmpeg_stream_process(output_rtsp_url, height, width, fps)

    # FLUSH queue before start, avoding delays
    print("Flushing Queue Before start")
    while not input_frame_queue.empty():
        input_frame_queue.get()

    # Begin Iteration:
    print("Begin Post Processing Iteration")
    while True:
        # Get Frame
        frame=input_frame_queue.get()
        
        # Resizing frame
        operating_frame = resize_frame(frame, scale_percent)
    
        # Getting Predictions
        y_hat = model.predict(
            operating_frame, 
            conf = confidence_threshold,
            classes = classes_IDs,
            device = gpu_device,
            verbose = False
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
            interest_line_coordinate_1,
            interest_line_coordinate_2,
            interest_color_rgb,
            text_size
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
                color_rgb, 
                line_size
            )
            
            # Draw Centre
            cv2.circle(
                operating_frame, 
                (center_x,center_y), 
                circle_radius,
                color_rgb,
                circle_thickness
            )
            
            # Write above bounding Box the name of class and Conf
            cv2.putText(
                img=operating_frame, 
                text=class_name+' - '+ str(round(confidence, 2)),
                org= (xmin,ymin-10), 
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                fontScale=font_scale, 
                color=color_rgb,
                thickness=text_size
            )
            
            ###############################################
            #################### BLoC #####################
            ###############################################
            
            counter_in, counter_in_classes, counter_out, counter_out_classes = function_dictionary[FUNCTION_INDICATOR](
                    (center_x, center_y), 
                    interest_line_coordinate_1, 
                    traffic_line_coordinate_1, 
                    class_ID,
                    counter_in, 
                    counter_in_classes,
                    counter_out, 
                    counter_out_classes,
                    offset
            )
            
            # Message Sending
            # message_queue.put(
            #     {
            #         "uuid": uuid.uuid4(),
            #         "label": class_name,
            #         "confidence": round(confidence, 2),
            #         "captured_timestamp": datetime.now(timezone.utc).isoformat(),
            #         "object_snippet": base64.b64encode(cv2.imencode('.jpg', frame[ymin:ymax, xmin:xmax])[1])
            #     }
            # )
                    
            ###############################################
            ###############################################
            ###############################################

        # Write the number of vehicles in\out
        cv2.putText(
            img=operating_frame, 
            text='N. Vehicles In', 
            org= detail_in_coordinate, # Coordinate of Text x,y
            fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
            fontScale=font_scale, 
            color=color_rgb,
            thickness=text_size
        )
        
        cv2.putText(
            img=operating_frame, 
            text='N. Vehicles Out', 
            org= detail_out_coordinate, # Coordinate of Text x,y
            fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
            fontScale=font_scale, 
            color=color_rgb,
            thickness=text_size
        )

        # Writing the counting of type of vehicles in the corners of frame 
        y_increment = 30
        for _ in classes_IDs:
            # IN
            cv2.putText(
                img=operating_frame, 
                text= f"{classes_names[_]} : {counter_in_classes[_]}", 
                org= (detail_in_coordinate[0], detail_in_coordinate[1] + y_increment), 
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
                fontScale=font_scale, 
                color=color_rgb,
                thickness=text_size
            )
            
            # OUT
            cv2.putText(
                img=operating_frame, 
                text= f"{classes_names[_]} : {counter_out_classes[_]}", 
                org= (detail_out_coordinate[0], detail_out_coordinate[1] + y_increment), 
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=font_scale, 
                color=color_rgb,
                thickness=text_size
            )
            y_increment +=30
        # Writing the number of vehicles in\out
        # IN
        cv2.putText(
            img=operating_frame, 
            text=f'In:{counter_in}', 
            org= total_in_coordinate,
            fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
            fontScale=font_scale*2, 
            color=interest_color_rgb,
            thickness=text_size        
        )
        # OUT
        cv2.putText(
            img=operating_frame, 
            text=f'Out:{counter_out}', 
            org= total_out_coordinate,
            fontFace=cv2.FONT_HERSHEY_TRIPLEX, 
            fontScale=font_scale*2, 
            color=interest_color_rgb,
            thickness=text_size
        )

        # Publish To RTSP
        ffmpeg_process.stdin.write(
            operating_frame.astype(np.uint8).tobytes()
        )

def message_function(input_message_queue, input_json):
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
    ###############################################
    ############# Reading Variables ###############
    ###############################################
    
    ###############################################
    
    ############ Configuration Setup ##############
    ###############################################
    
    # Initialize Influx DB
    client = influxdb_client.InfluxDBClient(
        url=INFLUX_DB_URL,
        username=INFLUX_DB_USERNAME, 
        password=INFLUX_DB_PASSWORD,
        org=INFLUX_DB_ORG
    )
    write_api = client.write_api(write_options=SYNCHRONOUS)
    
    ###############################################
    ############# Starting Iterator ###############
    ###############################################
    message_counter = 1
    while True:
        if input_message_queue.empty() !=True:
            # Get Frame
            operating_message = input_message_queue.get()
            
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
    