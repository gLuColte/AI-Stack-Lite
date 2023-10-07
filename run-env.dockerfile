#####################################
############ Image Pull #############
#####################################
# Pull Cuda Environment
FROM ai-stack-lite-base-1:latest

#####################################
########## Configuration ############
#####################################
# Expose Port
EXPOSE 5000

#####################################
############# File Copy #############
#####################################
# Copy all files into file Copy
ADD . /workspace/ai-stack-lite
# Change Directory
WORKDIR /workspace/ai-stack-lite

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
ENTRYPOINT ["sh", "./run.sh"]