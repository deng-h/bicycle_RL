#!/bin/bash

clear
echo "开始执行"

python enjoy.py \
--algo ppo \
--env BicycleMazeLidar-v0 \
--folder logs/ \
--exp-id 3 \
--env-kwargs gui:True \
--load-best \
--no-render \
--n-timesteps 50000
