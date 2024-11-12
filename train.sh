#!/bin/bash

clear
echo "开始执行"

python train.py \
--algo ppo \
--env BicycleMaze-v0 \
--conf-file ppo_config \
--tensorboard-log ./logs/tensorboard/ \
--vec-env subproc \
--progress


# python train.py \
# --algo ppo \
# --env BicycleMaze-v0 \
# --conf-file ppo_config \
# --trained-agent ./logs/ppo/BicycleMaze-v0_1/best_model.zip \
# --vec-env subproc 
# --progress