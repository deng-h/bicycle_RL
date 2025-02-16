#!/bin/bash

clear
echo "开始执行"

# 平衡
#python enjoy.py \
#--algo ppo \
#--env BicycleBalance-v0 \
#--folder logs/ \
#--exp-id 4 \
#--env-kwargs gui:True \
#--load-best \
#--no-render \
#--n-timesteps 10000 \
#--device cpu

# 导航
python enjoy.py \
--algo ppo \
--env BicycleMazeLidar5-v0 \
--folder logs/ \
--exp-id 2 \
--env-kwargs gui:True \
--load-best \
--no-render \
--device cpu \
--seed 2945312559 \
--n-timesteps 15000
