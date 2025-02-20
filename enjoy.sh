#!/bin/bash

clear
echo "开始执行"

# 导航
/home/chen/anaconda3/envs/denghang/bin/python enjoy.py \
--algo ppo \
--env ZBicycleNaviEnv-v0 \
--folder logs/ \
--exp-id 6 \
--env-kwargs gui:True \
--load-best \
--no-render \
--seed 42 \
--deterministic \
--device cpu \
--n-timesteps 1000000
