#!/bin/bash

clear
echo "开始执行"

# 平衡训练
#python train.py \
#--algo ppo \
#--env BicycleBalance-v0 \
#--conf-file ppo_config \
#--vec-env subproc \
#--progress \
#--device cpu
#--tensorboard-log ./logs/tensorboard/

python train.py \
--algo ppo \
--env BicycleMazeLidar2-v0 \
--conf-file ppo_config \
--vec-env subproc \
--progress \
--device cpu
--trained-agent ./logs/ppo/BicycleMazeLidar2-v0_3/best_model.zip

#--tensorboard-log ./logs/tensorboard/
