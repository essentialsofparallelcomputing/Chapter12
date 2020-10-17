#!/bin/sh
docker build -f Dockerfile.Ubuntu20.04 -t chapter12 .
#docker run --gpus all -it --entrypoint /bin/bash chapter12
docker run -it --device=/dev/kfd --device=/dev/dri --group-add video -it --entrypoint /bin/bash chapter12
