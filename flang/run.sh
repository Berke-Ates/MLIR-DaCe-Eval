#!/bin/bash

# Don't forget to start docker daemon: sudo dockerd

export DOCKER_BUILDKIT=1

# docker build -t flang . && \
docker run -it --rm -v $PWD/out:/home/out flang
