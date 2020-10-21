#!/bin/sh
docker build -f Dockerfile.OpenCL.Nvidia -t chapter12.OpenCL .
#docker run --rm --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,display,video,graphics -it --entrypoint /bin/bash chapter12
docker run --gpus all -it --entrypoint /bin/bash chapter12.OpenCL
