#!/bin/bash

# Don't forget to start docker daemon: sudo dockerd

export DOCKER_BUILDKIT=1

# docker build -t mlir_hlo_orig . && \
docker run -it --rm -v $PWD/../torch-mlir/out:/home/out mlir_hlo_orig
