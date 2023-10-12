
![AI Stack Lige](/markdown-images/main-logo.png)

## Background

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

## Architecture

This repository presents our proposed architecture designed to streamline monitoring and orchestration processes for containerized applications. The architecture is divided into four primary layers: Visualization, Databases, Modules, and Orchestration. It integrates powerful tools like Grafana, Prometheus, and InfluxDB for efficient data visualization and storage. On the module front, it incorporates cAdvisor, Node Exporter, and specialized AI inference modules for comprehensive data collection and processing. For orchestration, we propose a flexible approach, allowing users to choose between Docker Compose and Kubernetes, all running on the robust Ubuntu operating system. This architecture ensures efficient data flow, from raw metrics collection to insightful visualization, ensuring optimal performance and observability of your applications.

![Architecture](/markdown-images/architecture.png)

## TODOs

- [X] Local Inference Script Update
- [X] Live Inference Script Update - Fix Output stream Dimensions
- [X] Influx DB setup
- [X] Influx Python Local Inserter
- [ ] Live Count Update
- [ ] Live Inference Pictures with Data Insert
- [ ] GPU Device Monitoring
- [ ] Live Inference Speed / Accuracies Monitoring
- [ ] Mojo vs Python Test
- [ ] Kubernetes Setup
- [ ] Alert Manager Setup
- [ ] Cloud Deployment - Integration with Azure and AWS
- [ ] ReadMe Documentation - Grafana/Influx/cAdvisor/NodeExporter/AlertManager

## Setup

### Base Image Building

Currently there is only a single Base Image. First, build Base image:

```terminal
docker build -f ./build/base-env.dockerfile -t ai-stack-lite-base-1 .
```

After building the base image, build Run image:

```terminal
docker build -f run-env.dockerfile -t ai-stack-lite-run-1 .
```

In order to simulate real world scenario, a Camera Stream is needed, in this case, [MediaMTX](https://github.com/bluenviron/mediamtx) is used to assist. Build MediaMTX image:

```terminal
docker build -f ./mediamtx/emulator-env.dockerfile -t mediamtx-env-1 .
```

### Modular Token Key

Start by creating env files in the build directory:

```terminal
/build/build.env
```

Ensure the following variables are available in **build.env**:

```terminal
MOJO_KEY=<Your Key>
```

### Execution

For debugging purpose, you can run only Single Module interactively:

```terminal
docker run -it --gpus all -t ai-stack-lite-base-1:latest
```

Using Docker-Compose:

```terminal
docker-compose -f docker-compose.yml up
```

[NOT COMPLETED] Using Kubernetes (Ensure to have [minikube](https://github.com/kubernetes/minikube) installed):

```terminal
minikube start
kubectl 
```

## AI Inference

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

Output Grafana Visualization:

![Sample Grafana Visualization](./markdown-images/sample-visualization.png)

## Monitoring

The uses of off-the-shelf modules (Grafana, Prometheus, node-exporter, cadvisor) and setup to monitor Host and docker environments:

Node Exporter         |
:--------------------:|
![Node Exporter](/markdown-images/node-exporter.png)

cAdvisor              |
:--------------------:|
![cAdvisor](/markdown-images/cadvisor.png)

## Cloud Deployments

TBD

## Reference Sites

Following the links:

- [Enhancing AI Development with Mojo: Code Examples and Best Practices](https://artificialcorner.com/enhancing-ai-development-with-mojo-code-examples-and-best-practices-6341c3e66e15)
- [Get started with Mojo🔥](https://docs.modular.com/mojo/manual/get-started/index.html)
- [How to install Mojo🔥locally using Docker](https://medium.com/@1ce.mironov/how-to-install-mojo-locally-using-docker-5346bc23a9fe)
