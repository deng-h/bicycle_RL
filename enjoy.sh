#!/bin/bash

clear
echo "开始执行"

python enjoy.py \
--algo ppo \
--env BicycleBalance-v0 \
--folder logs/ \
--exp-id 4 \
--env-kwargs gui:True \
--load-best \
--no-render \
--n-timesteps 10000 \
--device cpu
