#!/bin/bash

# Don't forget to start docker daemon: sudo dockerd

export DOCKER_BUILDKIT=1

# docker build -t mlir_hlo . && \
docker run -it --rm -v $PWD/out:/home/out mlir_hlo
