# RTSP Simulator - Single Docker Version
# Table of Contents
- [RTSP Simulator - Single Docker Version](#rtsp-simulator---single-docker-version)
- [Table of Contents](#table-of-contents)
- [Introduction](#introduction)
- [Files](#files)
    - [When executed, 2 extra files will be generated:](#when-executed-2-extra-files-will-be-generated)
- [Build](#build)
- [Connecting to RTSP Paths](#connecting-to-rtsp-paths)
- [Sample IoT Edge Deployment Module Format](#sample-iot-edge-deployment-module-format)
- [Protocols](#protocols)
  - [Explaination](#explaination)
    - [Original](#original)
    - [Adjusted](#adjusted)
- [RunOnInit, RunOnDemand and RunOnReady](#runoninit-runondemand-and-runonready)



# Introduction 
This repository creates a docker that simulates RTSP through rtsp://localhost:8554/<custom-endpoint>
# Files
- **Dockerfile**: File to build the docker from alpine and rtsp-simple-server
- **\[template\]-rtsp-simple-server.yml**: A yaml file only contains the first half of configuration
- **run_module.sh**: Main execution script
- **rtsp-yml-adjustor.py**: Part of execution shell script, converting envrionmental variables to YAML for execution. Including downloading s3,
- **videos**: Directory for storing Videos
### When executed, 2 extra files will be generated:
- **generated_paths.yml**: This contains the second half of configuration, of the following format:
- **rtsp-simple-server.yml**: This is the merged YAML file that will be loaded by rtsp-simple-server executable

# Build
- Build Image via direct commit to DevOps
# Connecting to RTSP Paths
RTSP are located:
- **A sample video is loaded on**:
  - rtsp://\<Module-Name\>:8554/sample-video
- **All modules are of the format**:
  - rtsp://\<Module-Name\>:8554/\<endpoint-name\>
# Sample IoT Edge Deployment Module Format

```json
"rtsp-simulator": {
    "version": "1.0",
    "type": "docker",
    "status": "running",
    "restartPolicy": "always",
    "startupOrder": 0,
    "settings": {
        "image": "ailumacontainers.azurecr.io/ai-rtsp-simulator-module:10217-single-docker-rtsp-streamer-20220304.14-1d285b2469bd275d4a25c91c518d0d26b49ab673",
        "createOptions": "{\"HostConfig\":{\"PortBindings\":{\"8554/tcp\":[{\"HostPort\":\"8554\"}]}}}"
    },
    "env": {
        "lesion-rtsp": {
            "value": "{\"loop\": \"True\", \"s3-link\": \"s3://luma-deploy-bucket/test-videos/thomastown-lesion/video1-199feet.mkv\"}"
        },
        "hygiene-rtsp": {
            "value": "{\"loop\": \"True\", \"s3-link\": \"s3://lumachain-raw-dataset/temp-folder/meat_cut_ori.mp4\"}"
        },
        "hygiene-false-rtsp": {
            "value": "{\"loop\": \"False\", \"s3-link\": \"s3://lumachain-raw-dataset/temp-folder/meat_cut_ori.mp4\"}"
        },
        "AWSKeyID": {
            "value": "#{awsAccess}#"
        },
        "AWSSecretKey": {
            "value": "#{awsSecret}#"
        },
        "AWSRegion": {
        "value": "#{awsRegion}#"
        }
    }
}
```
# Protocols
## Explaination
Within the template, it is forced to use TCP connection. The setting can be adjusted at line 68, where original 3 were given, but removed to '\[tcp\]':
- UDP is the most performant, but doesn't work when there's a NAT/firewall between server and clients, and doesn't support encryption.
- UDP-multicast allows to save bandwidth when clients are all in the same LAN.
- TCP is the most versatile, and does support encryption.
### Original
```yaml
# The handshake is always performed with TCP.
protocols: [udp, multicast, tcp]
```
### Adjusted
```yaml
# The handshake is always performed with TCP.
protocols: [tcp]
```

# RunOnInit, RunOnDemand and RunOnReady
The YAML is using runOnInit, meaning it runs the stream when rtsp docker is initialized, however it can be changed to any of the following:
```yaml
###############################################
# Path parameters

# These settings are path-dependent, and the map key is the name of the path.
# It's possible to use regular expressions by using a tilde as prefix.
# For example, "~^(test1|test2)$" will match both "test1" and "test2".
# For example, "~^prefix" will match all paths that start with "prefix".
# The settings under the path "all" are applied to all paths that do not match
# another entry.
paths:
  all:
    # Source of the stream. This can be:
    # * publisher -> the stream is published by a RTSP or RTMP client
    # * rtsp://existing-url -> the stream is pulled from another RTSP server / camera
    # * rtsps://existing-url -> the stream is pulled from another RTSP server / camera with RTSPS
    # * rtmp://existing-url -> the stream is pulled from another RTMP server
    # * http://existing-url/stream.m3u8 -> the stream is pulled from another HLS server
    # * https://existing-url/stream.m3u8 -> the stream is pulled from another HLS server with HTTPS
    # * redirect -> the stream is provided by another path or server
    source: publisher

    # If the source is an RTSP or RTSPS URL, this is the protocol that will be used to
    # pull the stream. available values are "automatic", "udp", "multicast", "tcp".
    sourceProtocol: automatic

    # Tf the source is an RTSP or RTSPS URL, this allows to support sources that
    # don't provide server ports or use random server ports. This is a security issue
    # and must be used only when interacting with sources that require it.
    sourceAnyPortEnable: no

    # If the source is a RTSPS or HTTPS URL, and the source certificate is self-signed
    # or invalid, you can provide the fingerprint of the certificate in order to
    # validate it anyway. It can be obtained by running:
    # openssl s_client -connect source_ip:source_port </dev/null 2>/dev/null | sed -n '/BEGIN/,/END/p' > server.crt
    # openssl x509 -in server.crt -noout -fingerprint -sha256 | cut -d "=" -f2 | tr -d ':'
    sourceFingerprint:

    # If the source is an RTSP or RTMP URL, it will be pulled only when at least
    # one reader is connected, saving bandwidth.
    sourceOnDemand: no
    # If sourceOnDemand is "yes", readers will be put on hold until the source is
    # ready or until this amount of time has passed.
    sourceOnDemandStartTimeout: 10s
    # If sourceOnDemand is "yes", the source will be closed when there are no
    # readers connected and this amount of time has passed.
    sourceOnDemandCloseAfter: 10s

    # If the source is "redirect", this is the RTSP URL which clients will be
    # redirected to.
    sourceRedirect:

    # If the source is "publisher" and a client is publishing, do not allow another
    # client to disconnect the former and publish in its place.
    disablePublisherOverride: no

    # If the source is "publisher" and no one is publishing, redirect readers to this
    # path. It can be can be a relative path  (i.e. /otherstream) or an absolute RTSP URL.
    fallback:

    # Username required to publish.
    # SHA256-hashed values can be inserted with the "sha256:" prefix.
    publishUser:
    # Password required to publish.
    # SHA256-hashed values can be inserted with the "sha256:" prefix.
    publishPass:
    # IPs or networks (x.x.x.x/24) allowed to publish.
    publishIPs: []

    # Username required to read.
    # SHA256-hashed values can be inserted with the "sha256:" prefix.
    readUser:
    # password required to read.
    # SHA256-hashed values can be inserted with the "sha256:" prefix.
    readPass:
    # IPs or networks (x.x.x.x/24) allowed to read.
    readIPs: []

    # Command to run when this path is initialized.
    # This can be used to publish a stream and keep it always opened.
    # This is terminated with SIGINT when the program closes.
    # The following environment variables are available:
    # * RTSP_PATH: path name
    # * RTSP_PORT: server port
    # * G1, G2, ...: regular expression groups, if path name is
    #   a regular expression.
    runOnInit:
    # Restart the command if it exits suddenly.
    runOnInitRestart: no

    # Command to run when this path is requested.
    # This can be used to publish a stream on demand.
    # This is terminated with SIGINT when the path is not requested anymore.
    # The following environment variables are available:
    # * RTSP_PATH: path name
    # * RTSP_PORT: server port
    # * G1, G2, ...: regular expression groups, if path name is
    #   a regular expression.
    runOnDemand:
    # Restart the command if it exits suddenly.
    runOnDemandRestart: no
    # Readers will be put on hold until the runOnDemand command starts publishing
    # or until this amount of time has passed.
    runOnDemandStartTimeout: 10s
    # The command will be closed when there are no
    # readers connected and this amount of time has passed.
    runOnDemandCloseAfter: 10s

    # Command to run when the stream is ready to be read, whether it is
    # published by a client or pulled from a server / camera.
    # This is terminated with SIGINT when the stream is not ready anymore.
    # The following environment variables are available:
    # * RTSP_PATH: path name
    # * RTSP_PORT: server port
    # * G1, G2, ...: regular expression groups, if path name is
    #   a regular expression.
    runOnReady:
    # Restart the command if it exits suddenly.
    runOnReadyRestart: no

    # Command to run when a clients starts reading.
    # This is terminated with SIGINT when a client stops reading.
    # The following environment variables are available:
    # * RTSP_PATH: path name
    # * RTSP_PORT: server port
    # * G1, G2, ...: regular expression groups, if path name is
    #   a regular expression.
    runOnRead:
    # Restart the command if it exits suddenly.
    runOnReadRestart: no
```



