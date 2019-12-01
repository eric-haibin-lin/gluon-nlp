#!/bin/bash
CONTAINER_NAME=$1
CLUSTER_USER=$2
CONTAINER_REGISTRY=$3
DATA_PATH=/home/ec2-user/mxnet-data

docker kill $CONTAINER_NAME > /dev/null

set -ex

docker pull $CONTAINER_REGISTRY

nvidia-docker run \
    --shm-size=32g \
    --rm \
    --name $CONTAINER_NAME \
    --net=host --uts=host --ipc=host \
    --ulimit stack=67108864 --ulimit memlock=-1 \
    --ulimit nofile=8192:8192 \
    --security-opt seccomp=unconfined \
    -v $DATA_PATH:/home/$CLUSTER_USER/mxnet-data \
    -e FI_PROVIDER=\"efa\" \
    --device=/dev/infiniband/uverbs0 \
    --detach \
    -e NVIDIA_VISIBLE_DEVICES=all \
    $CONTAINER_REGISTRY

sleep 5
