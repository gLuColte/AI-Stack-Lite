# Modular-AI-Playground

Following the links:

- [Enhancing AI Development with Mojo: Code Examples and Best Practices](https://artificialcorner.com/enhancing-ai-development-with-mojo-code-examples-and-best-practices-6341c3e66e15)
- [Get started with Mojo🔥](https://docs.modular.com/mojo/manual/get-started/index.html)
- [How to install Mojo🔥locally using Docker](https://medium.com/@1ce.mironov/how-to-install-mojo-locally-using-docker-5346bc23a9fe)

## TODOs

- Local Inference Script Update
- Live Inference Script Update
- Count Update
- Influx DB setup
- Live Inference with Data Insert
- Live Inference Speed
- ReadMe Documentation, expand on this overall project

## Setup

Start by creating env files in the root level:

```terminal
/build/build.env
```

Ensure the following variables are available in **build.env**:

```terminal
MOJO_KEY=<Your Key>
```

## Build

Build Base image:

```terminal
docker build -f base-env.dockerfile -t mojo-pytorch-base-1 .
```

Build Run image:

```terminal
docker build -f run-env.dockerfile -t mojo-run-1 .
```

Build MediaMTX image:

```terminal
docker build -f /mediamtx-module/emulator-env.dockerfile -t mediamtx-env-1 .
```

## Execution

Run only Single Module interactively:

```terminal
docker run -it --gpus all -t mojo-run-1:latest
```

Run with Docker-Compose:

```terminal
docker-compose -f docker-compose.yml up
```


## Monitoring

In a general aspect of things, the understanding of the relationship inbetween are as follow:
- Grafana: Dashboard, mainly for Data visualization
- Prometheus + Alert Manager
- Node Exporter / Cadvisor + Reddis
