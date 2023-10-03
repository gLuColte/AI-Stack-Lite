# Modular-AI-Playground

In today's AI-driven landscape, real-time monitoring and visualization of AI inference results are paramount. While AI models churn through vast datasets and generate outputs, interpreting these results in real-time and ensuring the system's optimal health remains a challenge. Traditional monitoring tools might fall short in catering to the specific needs of AI workloads. Moreover, there's a growing need for local solutions that can simulate production-like environments without the overhead of cloud deployments, especially during the development and testing phases.

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

## Monitoring

## Problem Statement + Solution Created ( Portfolio )
## Describing competiencies and driving towards solution

## Raw Video + Inferenced Video > Readme

## COde used, SOlution You made
## What Video it is using and output




## Reference Sites

Following the links:

- [Enhancing AI Development with Mojo: Code Examples and Best Practices](https://artificialcorner.com/enhancing-ai-development-with-mojo-code-examples-and-best-practices-6341c3e66e15)
- [Get started with Mojo🔥](https://docs.modular.com/mojo/manual/get-started/index.html)
- [How to install Mojo🔥locally using Docker](https://medium.com/@1ce.mironov/how-to-install-mojo-locally-using-docker-5346bc23a9fe)