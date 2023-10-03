#####################################
############ Image Pull #############
#####################################
FROM bluenviron/mediamtx:latest as mediamtx
FROM alpine:3.12

#####################################
########## Configuration ############
#####################################
# Expose Port
EXPOSE 8554

#####################################
############# File Copy #############
#####################################
# Copy all files into file Copy
ADD ./mediamtx-module /workspace/mediamtx-module
# Change Directory
WORKDIR /workspace/mediamtx-module

#####################################
########### Installations ###########
#####################################
# Set non interactive
ENV DEBIAN_FRONTEND=noninteractive

# Installing Relevant Packages
# 1) FFmpeg
RUN apk add --no-cache ffmpeg

# 2) Python 
RUN apk add --no-cache \
        python3 \
        py3-pip \
    && pip3 install --upgrade pip \
    && pip3 install --no-cache-dir \
        awscli \
    && rm -rf /var/cache/apk/*

# Copy rtsp simple executable
COPY --from=mediamtx /mediamtx /

#####################################
############# Execute RUN ###########
#####################################
ENTRYPOINT ["sh", "run.sh" ]