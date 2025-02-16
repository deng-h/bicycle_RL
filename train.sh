#!/bin/bash

clear
echo "开始执行"

# 检测操作系统
OS=$(uname)

# 根据操作系统设置配置文件名称
if [ "$OS" = "Linux" ]; then
    CONF_FILE="ppo_config_linux"
else
    CONF_FILE="ppo_config"
fi

# 平衡训练
#python train.py \
#--algo ppo \
#--env BicycleBalance-v0 \
#--conf-file ppo_config \
#--vec-env subproc \
#--progress \
#--device cpu
#--tensorboard-log ./logs/tensorboard/

# BicycleMazeLidar2
#python train.py \
#--algo ppo \
#--env BicycleMazeLidar2-v0 \
#--conf-file "$CONF_FILE" \
#--vec-env subproc \
#--progress \
#--device cpu
#--trained-agent ./logs/ppo/BicycleMazeLidar2-v0_2/best_model.zip
#--tensorboard-log ./logs/tensorboard/

# BicycleMazeLidar3
#python train.py \
#--algo ppo \
#--env BicycleMazeLidar3-v0 \
#--conf-file "$CONF_FILE" \
#--vec-env subproc \
#--progress \
#--device cpu \
#--trained-agent ./logs/ppo/BicycleMazeLidar3-v0_5/best_model.zip

# BicycleMazeLidar5
python train.py \
--algo ppo \
--env BicycleMazeLidar5-v0 \
--conf-file "$CONF_FILE" \
--vec-env subproc \
--progress \
--seed 2945312559 \
--device cpu
#--optimize-hyperparameters \
#--n-jobs 2 \

#--trained-agent ./logs/ppo/BicycleMazeLidar4-v0_4/best_model.zip

# BicycleDengh-v0
#python train.py \
#--algo ppo \
#--env BicycleDengh-v0 \
#--conf-file "$CONF_FILE" \
#--vec-env subproc \
#--progress \
#--device cpu \
#--trained-agent ./logs/ppo/BicycleDengh-v0_2/best_model.zip