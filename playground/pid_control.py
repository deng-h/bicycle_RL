import time
import numpy as np
from bicycle_dengh.resources import bicycle
from simple_pid import PID
import gymnasium as gym
import pybullet as p
import csv

roll_angles = []
steps = []
step = 0

roll_angle_pid = PID(10, 0.5, 0, setpoint=0.0)
env = gym.make('BicycleMazeLidar-v0', gui=True)
obs, _ = env.reset()
flywheel_vel = []

for i in range(10000):
    bicycle_vel = p.readUserDebugParameter(env.bicycle_vel_param, physicsClientId=env.client)
    handlebar_angle = p.readUserDebugParameter(env.handlebar_angle_param, physicsClientId=env.client)

    roll_angle_control = roll_angle_pid(obs['obs'][0])
    flywheel_vel.append(obs['obs'][5])
    action = np.array([handlebar_angle, -1.0, -roll_angle_control], np.float32)
    obs, _, terminated, truncated, infos = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
        roll_angle_pid.reset()

env.close()
with open("PID飞轮数据.csv", mode='w', newline='') as file:
    # 创建一个 CSV 写入器对象
    writer = csv.writer(file)
    for value in flywheel_vel:
        writer.writerow([value])  # 每个值写成一行
