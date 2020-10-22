#!/bin/sh
docker build -f Dockerfile.DPCPP -t chapter12 .
#docker run --gpus all nvidia/cuda:11.0-base nvidia-smi
#docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,display,video,graphics -it --entrypoint /bin/bash chapter12
docker run --rm --gpus all -it --entrypoint /bin/bash chapter12
#docker run --device=/dev/dri --gpus all -it --entrypoint /bin/bash chapter12
#docker run -it --device=/dev/dri --group-add video -it --entrypoint /bin/bash chapter12
