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

python train.py \
--algo ppo \
--env BicycleMazeLidar2-v0 \
--conf-file "$CONF_FILE" \
--vec-env subproc \
--progress \
--device cpu \
--trained-agent ./logs/ppo/BicycleMazeLidar2-v0_4/best_model.zip

#--tensorboard-log ./logs/tensorboard/
