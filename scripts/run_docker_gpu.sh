#!/bin/bash
# Launch an experiment using the docker gpu image
cmd_line="$@"
echo "Executing in the docker (gpu image):"
echo $cmd_line

# Using new-style GPU argument
NVIDIA_ARG="--gpus all"

docker run -it ${NVIDIA_ARG} --rm --network host --ipc=host \
--mount src=$(pwd),target=/home/mamba/stable-baselines3,type=bind stablebaselines/stable-baselines3:latest \
bash -c "cd /home/mamba/stable-baselines3/ && $cmd_line"