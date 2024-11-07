#!/bin/bash

# Check if --gpus X is provided as argument
if [ "$1" == "--gpus" ]; then
    GPUS=$2
    shift 2
else
    GPUS="all"
fi

sudo docker run \
            -v ./gaussian-splatting:/content/gaussian-splatting \
            -v /datassd/jnazarenus/datasets:/data \
            -v ./bash_history.txt:/root/.bash_history \
            --rm \
            --gpus $GPUS \
            --shm-size=32gb \
            --net=host \
            -it \
            nazarenus/gaussians-fisheye:0.2