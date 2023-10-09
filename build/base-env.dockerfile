#####################################
############ Image Pull #############
#####################################
# Pull Cuda Environment
FROM pytorch/pytorch:latest

#####################################
############# File Copy #############
#####################################
# ! : Need to figure a way around, at the moment its copy twice
# Copy Build Scripts
ADD build ./build
# Work Dir
WORKDIR /workspace/build

# #####################################
# ########### Installations ###########
# #####################################
# Set non interactive
ENV DEBIAN_FRONTEND=noninteractive

# Update
RUN apt-get -y update

# Install Miscellanous Packages
RUN apt-get install -y \
    curl 

# Install Miscellanous Packages for Run
RUN apt-get install -y \
    nano \
    figlet \
    toilet 

# Install Python Package
RUN apt-get install -y \
    git \
    python3-pip \
    python3-opencv \
    libglib2.0-0

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install PyTorch and torchvision
RUN pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/cu111/torch_stable.html

# Install Python related requirements
RUN python3 -m pip install -r /workspace/build/python-requirements.txt

# Install MOJO with Secret
ENTRYPOINT ["sh", "/workspace/build/build.sh"]