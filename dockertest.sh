#!/bin/sh
docker build --no-cache -t chapter12 .
#docker run -it --entrypoint /bin/bash chapter12
docker build --no-cache -f Dockerfile.Ubuntu20.04 -t chapter12 .
#docker run -it --entrypoint /bin/bash chapter12
docker build --no-cache -f Dockerfile.debian -t chapter12 .
#docker run -it --entrypoint /bin/bash chapter12
