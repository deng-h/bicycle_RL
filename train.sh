#!/bin/bash

clear
echo "开始执行"

python train.py \
--algo ppo \
--env BicycleMazeLidar2-v0 \
--conf-file ppo_config \
--vec-env subproc \
--progress
#--tensorboard-log ./logs/tensorboard/


#python train.py \
#--algo ppo \
#--env BicycleMazeLidar-v0 \
#--conf-file ppo_config \
#--vec-env subproc \
#--progress \
#--trained-agent ./logs/ppo/BicycleMazeLidar-v0_1/best_model.zip
#--tensorboard-log ./logs/tensorboard/
