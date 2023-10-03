#######################################################################################################################
################################################ Libraries and Set up #################################################
#######################################################################################################################

import os
import glob
import yaml
import json
import datetime

#######################################################################################################################
##################################################### Parameters ######################################################
#######################################################################################################################

SAMPLE_VIDEO_PATH = "sample-inputs/videos/"

#######################################################################################################################
##################################################### Functions #######################################################
#######################################################################################################################


# Generate another Yaml file containing only path
def path_yaml_generator(input_rtsp_dict: dict, output_path:str) -> None:   
    """
    # Sample Structure
    # paths:
    #   sample-video:
    #     runOnInit: ffmpeg -re -stream_loop -1 -i sample.mp4 -c copy -f rtsp rtsp://0.0.0.0:$RTSP_PORT/$RTSP_PATH
    #     runOnInitRestart: no
    #   live-stream-1:
    #     runOnInitRestart: no
    """
    # Initialize writing Dictionary
    operating_dict = {}
    # Iterate
    for rtsp_name, rtsp_endpoint in input_rtsp_dict.items():
        # Falsify runOnInitRestart
        operating_dict[rtsp_endpoint] = {
            'runOnInitRestart': False
        }
        if 'sample' in rtsp_name:
            operating_dict[rtsp_endpoint]['runOnInit'] = f'ffmpeg -re -stream_loop -1 -i {SAMPLE_VIDEO_PATH+rtsp_name.split("-")[-1]} -c copy -f rtsp rtsp://localhost:$RTSP_PORT/{rtsp_endpoint}'
        
    # Dump json to yaml
    yaml.safe_dump({'paths': operating_dict}, open(output_path, 'w'), default_flow_style=False)

# Merge 2 yaml files
def yaml_merge(file_path1, file_path2, output_path):

    # read both yaml files as Dictionaries
    data1 = yaml.safe_load(open(file_path1, 'r'))
    data2 = yaml.safe_load(open(file_path2, 'r'))

    # Merge the dictionaries
    data1.update(data2) # Feel free to reverse the order of data1 and data2 and then update the dumper below
 
    # Write the merged dictionary to a new file
    with open(output_path, 'w') as yaml_output:
      yaml.safe_dump(data1, yaml_output, default_flow_style=False)


#######################################################################################################################
####################################################### Main ##########################################################
#######################################################################################################################

if __name__ == '__main__': 

    # 1) Load Environmental Variables
    rtsp_dict = {}
    for name, value in sorted(os.environ.items()):
        if 'rtsp' in name:
            rtsp_dict[name] = value
    print("[{}] INFO: Loaded RTSP Environmental Variables:\n{}".format(datetime.datetime.now(), rtsp_dict))

    # 2) Generate Path Yaml File
    paths_yaml_output_path = "generated_paths.yml"
    path_yaml_generator(input_rtsp_dict=rtsp_dict, output_path=paths_yaml_output_path)
    print("[{}] INFO: Generated Paths Yaml:\n{}".format(datetime.datetime.now(), paths_yaml_output_path))

    # 3) Combine Template Yaml and Path Yaml
    output_yaml_path = "mediamtx.yml"
    yaml_merge(file_path1='template.yml', file_path2=paths_yaml_output_path, output_path=output_yaml_path)
    print("[{}] INFO: Combined two Yamls to 1:\n{}".format(datetime.datetime.now(), output_yaml_path))