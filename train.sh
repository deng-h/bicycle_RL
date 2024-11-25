#!/bin/bash

clear
echo "开始执行"

python train.py \
--algo ppo \
--env BicycleMazeLidar-v0 \
--conf-file ppo_config \
--tensorboard-log ./logs/tensorboard/ \
--vec-env subproc \
--progress


#python train.py \
#--algo ppo \
#--env BicycleMazeLidar-v0 \
#--conf-file ppo_config \
#--tensorboard-log ./logs/tensorboard/ \
#--vec-env subproc \
#--progress \
#--trained-agent ./logs/ppo/BicycleMazeLidar-v0_2/best_model.zip
