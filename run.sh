#!/bin/bash

cd ~/denghang/bicycle-rl
echo "开始执行"
python train.py \
--algo ppo \
--env BicycleMaze-v0 \
--conf-file ./hyperparams/bicycle.yml