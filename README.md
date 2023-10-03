# Modular-AI-Playground

In the realm of artificial intelligence, the process of inference involves deploying trained models to make predictions on new data. As these inferences are made, especially in real-time scenarios, it becomes imperative to monitor the results for accuracy, anomalies, and performance. Monitoring in real-time ensures that the AI system is functioning optimally and provides actionable insights. Visualization complements monitoring by offering a graphical representation of the AI's performance metrics, making it easier for stakeholders to understand and act upon.

However, setting up such a system in a cloud environment directly can be resource-intensive and might not be ideal during the initial development and testing phases. This is where the concept of a localized system comes into play. By creating a local environment that mirrors the cloud, developers can test, monitor, and visualize AI inferences without the overheads of a full-scale cloud deployment.

Furthermore, with the rise of container technologies like Docker, it's now possible to package the AI module, monitoring tools, and visualization dashboards into isolated containers. This ensures consistency, scalability, and portability across different stages of development. Orchestration, using systems like Kubernetes, automates the deployment, scaling, and management of these containers, simulating a cloud-like environment on a local setup.

## Problem Statement

We need a solution that allows for real-time AI inference monitoring and visualization within a local, containerized environment, ensuring seamless scalability and transition to cloud deployments.

## Problem 
The challenge lies in developing a localized, containerized solution that:

1. **Runs AI Inference**: Incorporates a standalone AI module adept at processing data and generating time-series results.
2. **Stores and Retrieves Results**: An efficient mechanism to store time-series AI outputs and ensure swift data retrieval.
3. **Visualizes Data**: A dynamic visualization tool that offers real-time insights into the AI's performance, aiding stakeholders in decision-making.
4. **Monitors System Health**: A robust monitoring system that provides a holistic view of all components, from AI processing to data storage.
5. **Orchestrates Workloads**: Utilizes orchestration tools to manage, scale, and automate tasks, ensuring the local environment closely simulates cloud deployments.

## Objective

To design and implement a solution that seamlessly integrates an AI module, efficient data storage mechanisms, dynamic visualization tools, and comprehensive monitoring systems. This solution should be containerized and orchestrated, allowing for easy deployment, scaling, and management, all while simulating a cloud-like environment locally.

## Local Simulation
The solution should be deployable on local machines using containerization tools like Docker and orchestrated using platforms like Docker-Compose/Kubernetes. This setup aims to:

- Facilitate rapid prototyping and testing in a controlled environment.
- Simulate real-world cloud scenarios and workloads.
- Offer efficient debugging and troubleshooting capabilities.

## Constraints
- The AI module should remain modular and independent for straightforward updates and modifications.
- Data persistence must be ensured, even in the event of container failures.
- Real-time visualization capabilities should allow for specific time interval analyses.
- Comprehensive monitoring should cover all components, offering timely alerts for any anomalies.

## TODOs
- [X] Local Inference Script Update
- [X] Live Inference Script Update - Fix Output stream Dimensions
- [ ] Live Count Update
- [X] Influx DB setup
- [ ] Live Inference with Data Insert
- [ ] Live Inference Speed
- [ ] GPU Device Monitoring
- [ ] ReadMe Documentation, expand on this overall project
- [ ] Kubernetes Setup
- [ ] Cloud Deployment

## Setup

Start by creating env files in the build directory:

```terminal
/build/build.env
```

Ensure the following variables are available in **build.env**:

```terminal
MOJO_KEY=<Your Key>
```

## Build

Currently there is only a single Base Image. First, build Base image:

```terminal
docker build -f base-env.dockerfile -t mojo-pytorch-base-1 .
```

After building the base image, build Run image:

```terminal
docker build -f run-env.dockerfile -t mojo-run-1 .
```

In order to simulate real world scenario, a Camera Stream is needed, in this case, [MediaMTX](https://github.com/bluenviron/mediamtx) is used to assist. Build MediaMTX image:

```terminal
docker build -f /mediamtx-module/emulator-env.dockerfile -t mediamtx-env-1 .
```

## Execution

For debugging purpose, you can run only Single Module interactively:

```terminal
docker run -it --gpus all -t mojo-run-1:latest
```

For overall setup, please run with Docker-Compose:

```terminal
docker-compose -f docker-compose.yml up
```

## AI Inference

As an illustration, following shows the configuration of an emulator module and 2 Live Inference Module:
```docker-compose
services:
  ####################################
  ########## MediaMTX Module #########
  ####################################
  emulator-module:
    image: mediamtx-env-1:latest
    ports:
      - 8554:8554/tcp
    environment:
      - rtsp-live-stream-1=live-1
      - rtsp-live-stream-2=live-2
      - rtsp-live-stream-3=live-3
      - rtsp-sample-MOT1608raw.mp4=sample-1
      - rtsp-sample-MOT1602raw.mp4=sample-2
  ####################################
  ########### Python Module ##########
  ####################################
  python-module-1:
    image: mojo-run-1:latest
    ports:
      - 8001:5000/tcp
    environment:
      - RUN_TYPE=python
      - RUN_SCRIPT_PATH=apps/python/gpu-live-inference-keypoint.py
      - MODEL_PATH=yolov8n-pose.pt
      - VISUALIZATION=0
      - RTSP_INPUT=rtsp://emulator-module:8554/sample-1
      - RTSP_OUTPUT=rtsp://emulator-module:8554/live-1
    # Deploy on GPU
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
  python-module-2:
    image: mojo-run-1:latest
    ports:
      - 8002:5000/tcp
    environment:
      - RUN_TYPE=python
      - RUN_SCRIPT_PATH=apps/python/gpu-live-inference-keypoint.py
      - MODEL_PATH=yolov8n-pose.pt
      - VISUALIZATION=1
      - RTSP_INPUT=rtsp://emulator-module:8554/sample-2
      - RTSP_OUTPUT=rtsp://emulator-module:8554/live-2
    # Deploy on GPU
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

```

The emulator module contains 5 main streams, 2 replays sample footage (MOT1608raw.mp4 and MOT1602raw.mp4) recursively, and 3 live stream path opening wait for publishing. 

The python modules individually takes in the given $RTSP_INPUT and publish to $RTSP_OUTPUT based on given configurations: 
- $RUN_TYPE = Language to execute
- $RUN_SCRIPT = Script to execute
- $MODEL_PATH = Model to be used
- $VISUALIZATION = Boolean

As an example, you will see a similar input and output to the following:
Raw Video             |  Inferenced Video
:--------------------:|:--------------------:
![Raw Video](/markdown-images/MOT1602raw.gif)  |  ![Inferenced Video](/markdown-images/yolov8n-poseMOT1602raw.gif)

## Monitoring

In terms of Monitoring, following reference links, the docker-compose is setup as below:
```docker-compose
  # Expose Host Machine Hardware Metrics 
  node-exporter:
    image: prom/node-exporter
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/rootfs:ro
    command:
      - "--path.procfs=/host/proc"
      - "--path.sysfs=/host/sys"
      - --collector.filesystem.ignored-mount-points
      - "^/(sys|proc|dev|host|etc|rootfs/var/lib/docker/containers|rootfs/var/lib/docker/overlay2|rootfs/run/docker/netns|rootfs/var/lib/docker/aufs)($$|/)"
    ports:
      - 9100:9100
    restart: always
    deploy:
      mode: global

  # Log Scrapper - a system to collect and process metrics, not an event logging system
  prometheus:
    image: prom/prometheus
    restart: always
    volumes:
      - ./prometheus:/etc/prometheus/
      - prometheus_data:/prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
      - "--web.console.libraries=/usr/share/prometheus/console_libraries"
      - "--web.console.templates=/usr/share/prometheus/consoles"
    ports:
      - 9090:9090
    links:
      - cadvisor:cadvisor
      - alertmanager:alertmanager
    depends_on:
      - cadvisor

  # Dashboard
  grafana:
    image: grafana/grafana
    user: "472"
    restart: always
    environment:
      GF_INSTALL_PLUGINS: "grafana-clock-panel,grafana-simple-json-datasource"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./grafana/provisioning/:/etc/grafana/provisioning/
    env_file:
      - ./grafana/config.monitoring
    ports:
      - 3000:3000
    depends_on:
      - prometheus

  # Container Advisor, is an open-source tool developed by Google to monitor containers
  cadvisor:
    image: gcr.io/cadvisor/cadvisor
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:rw
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
    ports:
      - 8080:8080
    restart: always
    deploy:
      mode: global
  # cAdvisor depends on Reddis
  redis: 
    image: redis:latest 
    container_name: redis 
    ports: 
      - 6379:6379 
```

The uses of off-the-shelf modules (Grafana, Prometheus, node-exporter, cadvisor) and setup to monitor Host and docker environments:

Node Exporter         | 
:--------------------:|
![Node Exporter](/markdown-images/node-exporter.png)

cAdvisor              | 
:--------------------:|
![cAdvisor](/markdown-images/cadvisor.png)


## Reference Sites

Following the links:

- [Enhancing AI Development with Mojo: Code Examples and Best Practices](https://artificialcorner.com/enhancing-ai-development-with-mojo-code-examples-and-best-practices-6341c3e66e15)
- [Get started with Mojo🔥](https://docs.modular.com/mojo/manual/get-started/index.html)
- [How to install Mojo🔥locally using Docker](https://medium.com/@1ce.mironov/how-to-install-mojo-locally-using-docker-5346bc23a9fe)