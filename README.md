# Modular-AI-Playground

In today's AI-driven landscape, real-time monitoring and visualization of AI inference results are essential. As AI models process vast datasets and produce outputs, the ability to interpret these results in real-time and ensure the system's health is crucial. While traditional monitoring tools serve many purposes, there's a unique set of requirements for AI workloads that might necessitate specialized solutions. Embracing containerization and orchestration techniques, there's a growing emphasis on local solutions that can act as testing beds, simulating cloud-like production environments. This approach, leveraging technologies like Docker and Kubernetes, is especially valuable during the development and testing phases, ensuring a seamless transition to cloud deployments.

## Problem Statement

Develop a local, containerized solution that:
- Runs AI Inference: The system should incorporate an independent AI module capable of processing data and producing time-series results.
- Stores Results: Given the time-series nature of the AI outputs, an efficient database optimized for time-series data storage is essential.
- Visualizes Data: A real-time visualization tool is required to interpret the AI's results, enabling stakeholders to gauge its performance and derive actionable insights.
- Monitors System Health: With the interconnectedness of components, from the AI module to the database, continuous monitoring of the system's health and performance is crucial.

## Objective
```
Design and implement a solution that integrates an AI module, InfluxDB for time-series data storage, Grafana for data visualization, and Prometheus for system monitoring. Crucially, the entire solution should be containerized using docker-compose to allow developers and testers to simulate the runtime environment locally, ensuring a seamless transition from development to production.

PLUS: 
- Comparing Modular VS Python Run Speed by running CPU intensive operations
- Setup Simple YOLO Training Commands
```

## Local Simulation using Docker-Compose
The solution should be easily deployable on any local machine using docker-compose. This local setup aims to replicate the production environment, allowing for:
- Rapid prototyping and testing without cloud overhead.
- Simulating real-world scenarios and workloads.
- Efficient debugging in an isolated, controlled environment.

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