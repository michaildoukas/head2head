#!/bin/bash

ORG='head2head'
IMAGE='vulkan-base'
IMAGE_FEATURE='cudagl'
REPOSITORY="$ORG/$IMAGE/$IMAGE_FEATURE"
BUILD_VERSION=${1:-"1.1.108.0"}
CUDA_MAJOR_VERSION='9.2'
USER='happyuser'
CODE_NAME='xenial'
TAG="$BUILD_VERSION-$CUDA_MAJOR_VERSION-$CODE_NAME"

# run container

xhost +local:root

if [ -c /dev/video0 ]; then
    printf '%s\n' "webcam connected"
    docker run -it \
      --runtime=nvidia \
      -v $(pwd):/home/$USER/head2head \
      --workdir=/home/$USER/ \
      --shm-size=1g \
      --device /dev/video0 \
      --rm \
      --name ${IMAGE}-${TAG} \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      -e DISPLAY=unix$DISPLAY \
      ${REPOSITORY}:${TAG} bash
else
    printf '%s\n' "webcam not connected"
    docker run -it \
      --runtime=nvidia \
      -v $(pwd):/home/$USER/head2head \
      --workdir=/home/$USER/ \
      --shm-size=1g \
      --rm \
      --name ${IMAGE}-${TAG} \
      -v /tmp/.X11-unix:/tmp/.X11-unix \
      -e DISPLAY=unix$DISPLAY \
      ${REPOSITORY}:${TAG} bash
fi

xhost -local:root
