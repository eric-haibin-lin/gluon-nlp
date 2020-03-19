source parse_yaml.sh
CONFIG=$(parse_yaml phase1-256.config)
eval $CONFIG
set -e

docker pull $CONTAINER_REGISTRY
docker kill $CONTAINER_NAME || true
docker rm $CONTAINER_NAME -f || true

#    -e FI_PROVIDER=\"efa\" \
#    --device=/dev/infiniband/uverbs0 \

nvidia-docker run \
    --shm-size=32g --rm \
    --name $CONTAINER_NAME \
    --net=host --uts=host --ipc=host \
    --ulimit stack=67108864 --ulimit memlock=-1 \
    --ulimit nofile=8192:8192 \
    --security-opt seccomp=unconfined \
    -v $CONTAINER_SHARED_FS:/data \
    --detach \
    -e NVIDIA_VISIBLE_DEVICES=all \
    $CONTAINER_REGISTRY
