##########################################################
####################### Libraries ########################
##########################################################
from ultralytics import YOLO
import numpy as np
import subprocess
import cv2
import math
import os
import time

##########################################################
###################### Parameters ########################
##########################################################

# environment:
#     - RUN_TYPE=python
#     - RUN_SCRIPT_PATH=apps/python/gpu-live-inference-bounding.py
#     - MODEL_PATH=yolo-Weights/yolov8n.pt
#     - VISUALIZATION=0
#     - RTSP_INPUT=rtsp://emulator-module:8554/sample-1 
#     - RTSP_OUTPUT=rtsp://emulator-module:8554/live-1

##########################################################
####################### Functions ########################
##########################################################

def open_ffmpeg_stream_process(stream_path):
    args = (
        "ffmpeg -re -f rawvideo -pix_fmt "
        "rgb24 -s 1920x1080 -i pipe:0 -pix_fmt yuvj420p "
        f"-f rtsp {stream_path}"
    ).split()
    return subprocess.Popen(args, stdin=subprocess.PIPE)


##########################################################
######################### Main ###########################
##########################################################
if __name__ == "__main__":
    # Initialize Model
    model_path=os.environ['MODEL_PATH']
    model = YOLO(model_path)
    # object classes
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                "teddy bear", "hair drier", "toothbrush"
                ]
    # RUN
    input_stream=os.environ['RTSP_INPUT']
    output_stream=os.environ['RTSP_OUTPUT']
    visualization=bool(int(os.environ['VISUALIZATION']))
    ffmpeg_process = open_ffmpeg_stream_process(output_stream)
    capture = cv2.VideoCapture(input_stream)    
    while True:
        success, img = capture.read()
        if not success:
            print("Failed to grab")
            time.sleep(1)
            continue
        
        results = model(img, stream=True)
        if visualization:
            # coordinates
            for r in results:
                boxes = r.boxes

                for box in boxes:
                    # bounding box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

                    # put box in cam
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                    # confidence
                    confidence = math.ceil((box.conf[0]*100))/100
                    print("Confidence --->",confidence)

                    # class name
                    cls = int(box.cls[0])
                    print("Class name -->", classNames[cls])

                    # object details
                    org = [x1, y1]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    fontScale = 1
                    color = (255, 0, 0)
                    thickness = 2
                    cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

        ffmpeg_process.stdin.write(img.astype(np.uint8).tobytes())
        
    capture.release()
    ffmpeg_process.stdin.close()
    ffmpeg_process.wait()
