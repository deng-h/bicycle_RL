#!/bin/bash

clear
echo "开始执行"


#/home/chen/anaconda3/envs/denghang/bin/python enjoy.py \
python enjoy.py \
--algo ppo \
--env ZBicycleBalanceEnv-v0 \
--folder logs/ \
--exp-id 15 \
--env-kwargs gui:True \
--load-best \
--no-render \
--seed 42 \
--deterministic \
--device cpu \
--n-timesteps 1100
