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

#/home/chen/anaconda3/envs/denghang/bin/python train.py \
python train.py \
--algo ppo \
--env ZBicycleBalanceEnv-v0 \
--conf-file "$CONF_FILE" \
--vec-env subproc \
--progress \
--seed 42 \
--device cpu 
# --trained-agent ./logs/ppo/ZBicycleNaviEnv-v0_7/best_model.zip
