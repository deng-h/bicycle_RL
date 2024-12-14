import time
import numpy as np
from bicycle_dengh.resources import bicycle
from simple_pid import PID
import gymnasium as gym
import pybullet as p

env = gym.make('BicycleMazeLidar2-v0', gui=True)
obs, _ = env.reset()

for i in range(10000):
    bicycle_vel = p.readUserDebugParameter(env.bicycle_vel_param, physicsClientId=env.client)
    handlebar_angle = p.readUserDebugParameter(env.handlebar_angle_param, physicsClientId=env.client)

    action = np.array([handlebar_angle, -0.5], np.float32)
    obs, _, terminated, truncated, infos = env.step(action)

    if terminated or truncated:
        obs, _ = env.reset()

env.close()
