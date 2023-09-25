#!/bin/bash
#####################################
####### Initialize Build Env ########
#####################################

# ! : Can be Commented out if using docker-compose
set -a 
source /workspace/build/build.env
set +a

#####################################
######### Install Modular ###########
#####################################
curl https://get.modular.com | \
  MODULAR_AUTH=$MOJO_KEY \
  sh -
modular install mojo

#####################################
########### Modular Install #########
#####################################
# Set Modular Path
echo 'export MODULAR_HOME="$HOME/.modular"' >> ~/.bashrc
echo 'export PATH="$MODULAR_HOME/pkg/packages.modular.com_mojo/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
