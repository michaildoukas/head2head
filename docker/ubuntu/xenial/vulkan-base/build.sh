#!/bin/sh

ORG='head2head'
IMAGE='vulkan-base'
IMAGE_FEATURE='cudagl'
REPOSITORY="$ORG/$IMAGE/$IMAGE_FEATURE"
BUILD_VERSION=${1:-"1.1.108.0"}
CUDA_MAJOR_VERSION='9.2'
CUDNN_VERSION='7.6.1.34'
CONDA_PYTHON_VERSION='3'
CONDA_BASE_PACKAGE='anaconda'
CONDA_VERSION='2020.02'
LLVM_VERSION='8'
NVIDIA_DRIVER_VERSION='430.26'
VULKAN_SDK_VERSION="$BUILD_VERSION"
USER='developer'
USER_ID='1000'
CODE_NAME='xenial'
TAG="$BUILD_VERSION-$CUDA_MAJOR_VERSION-$CODE_NAME"

# use tar to dereference the symbolic links from the current directory,
# and then pipe them all to the docker build - command

tar -czh . | docker build - \
  --build-arg REPOSITORY=nvidia/cudagl \
  --build-arg TAG="$CUDA_MAJOR_VERSION-devel-ubuntu16.04" \
  --build-arg BUILD_VERSION=$BUILD_VERSION \
  --build-arg CUDA_MAJOR_VERSION=$CUDA_MAJOR_VERSION \
  --build-arg CUDNN_VERSION=$CUDNN_VERSION \
  --build-arg CONDA_PYTHON_VERSION=$CONDA_PYTHON_VERSION \
  --build-arg CONDA_BASE_PACKAGE=$CONDA_BASE_PACKAGE \
  --build-arg CONDA_VERSION=$CONDA_VERSION \
  --build-arg LLVM_VERSION=$LLVM_VERSION \
  --build-arg NVIDIA_DRIVER_VERSION=$NVIDIA_DRIVER_VERSION \
  --build-arg VULKAN_SDK_VERSION=$VULKAN_SDK_VERSION \
  --build-arg USER=$USER \
  --build-arg UID=$USER_ID \
  --tag=${REPOSITORY}:${TAG} \
  --file docker/ubuntu/xenial/vulkan-base/Dockerfile
