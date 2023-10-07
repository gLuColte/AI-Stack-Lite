#!/bin/bash
#####################################
########### Initialize Sh ###########
#####################################
# "tput: No value for $TERM and no -T specified " Error Work Around
export TERM="xterm-256color"

# Define Banner Function
banner()
{
  echo "+------------------------------------------+"
  printf "| %-40s |\n" "`date`"
  echo "|                                          |"
  printf "|`tput bold` %-40s `tput sgr0`|\n" "$@"
  echo "+------------------------------------------+"
}

#####################################
################ Start ##############
#####################################
# Start
figlet -t "AI Stack Lite"

#####################################
########### Execute Scripts #########
#####################################
# Run Scriptj
banner "Starting Scripts"
# Change Work Dir to Apps
cd /workspace/ai-stack-lite
echo "Current working Directory:" $(pwd)
echo "RUN_SCRIPT_PATH:" $RUN_SCRIPT_PATH
echo "MODEL_PATH:" $MODEL_PATH
echo "VISUALIZATION:" $VISUALIZATION
echo "RTSP_INPUT:" $RTSP_INPUT
echo "RTSP_OUTPUT:" $RTSP_OUTPUT

# Run Scripts
python3 $RUN_SCRIPT_PATH

#####################################
############# Finish Note ###########
#####################################
# Finish
banner "Finished"