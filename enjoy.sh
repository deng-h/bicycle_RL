#!/bin/bash

cd ~/denghang/bicycle-rl
clear
echo "开始执行"

python enjoy.py \
--algo ppo \
--env BicycleMaze-v0 \
--folder logs/ \
--exp-id 7 \
--env-kwargs gui:True \
--load-best \
--no-render \
--n-timesteps 5000
