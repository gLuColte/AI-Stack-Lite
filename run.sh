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
######## Envrionment Variables ######
#####################################
# Variables
banner "Envrionment Variables"
# Change Work Dir to Apps
cd /workspace/ai-stack-lite
printenv

#####################################
########### Execute Scripts #########
#####################################
# Run Script
banner "Execute Script"
# Run Scripts
python3 $RUN_SCRIPT_PATH

#####################################
############# Finish Note ###########
#####################################
# Finish
banner "Finished"