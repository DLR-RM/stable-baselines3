#!/bin/bash

CPU_PARENT=mambaorg/micromamba:2.0-ubuntu24.04
GPU_PARENT=mambaorg/micromamba:2.0-cuda12.6.3-ubuntu24.04

TAG=stablebaselines/stable-baselines3
VERSION=$(cat ./stable_baselines3/version.txt)

if [[ ${USE_GPU} == "True" ]]; then
  PARENT=${GPU_PARENT}
  PYTORCH_DEPS="https://download.pytorch.org/whl/cu126"
else
  PARENT=${CPU_PARENT}
  PYTORCH_DEPS="https://download.pytorch.org/whl/cpu"
  TAG="${TAG}-cpu"
fi

echo "docker build --build-arg PARENT_IMAGE=${PARENT} --build-arg PYTORCH_DEPS=${PYTORCH_DEPS} -t ${TAG}:${VERSION} ."
docker build --build-arg PARENT_IMAGE=${PARENT} --build-arg PYTORCH_DEPS=${PYTORCH_DEPS} -t ${TAG}:${VERSION} .
docker tag ${TAG}:${VERSION} ${TAG}:latest

if [[ ${RELEASE} == "True" ]]; then
  docker push ${TAG}:${VERSION}
  docker push ${TAG}:latest
fi
