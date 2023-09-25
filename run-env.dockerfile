#####################################
############ Image Pull #############
#####################################
# Pull Cuda Environment
FROM mojo-pytorch-base-1:latest

#####################################
########## Configuration ############
#####################################
# Expose Port
EXPOSE 5000

#####################################
############# File Copy #############
#####################################
# Copy all files into file Copy
ADD . /workspace/modular-ai-playground
# Change Directory
WORKDIR /workspace/modular-ai-playground

# ! : Removing copied twice build
RUN rm -Rf ./build

#####################################
########### Installations ###########
#####################################
# Set non interactive
ENV DEBIAN_FRONTEND=noninteractive

#####################################
############# Execute RUN ###########
#####################################
ENTRYPOINT [ "./run.sh" ]