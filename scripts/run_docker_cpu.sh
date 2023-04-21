#!/bin/bash
# Launch an experiment using the docker cpu image

cmd_line="$@"

echo "Executing in the docker (cpu image):"
echo $cmd_line

docker run -it --rm --network host --ipc=host \
 --mount src=$(pwd),target=/home/mamba/stable-baselines3,type=bind stablebaselines/stable-baselines3-cpu:latest \
  bash -c "cd /home/mamba/stable-baselines3/ && $cmd_line"
