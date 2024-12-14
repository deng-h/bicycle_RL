import time
import numpy as np
from bicycle_dengh.resources import bicycle
from simple_pid import PID
import gymnasium as gym
import pybullet as p

roll_angles = []
steps = []
step = 0

roll_angle_pid = PID(10, 0, 0, setpoint=0.0)
env = gym.make('BicycleMazeLidar-v0', gui=True)
obs, _ = env.reset()

while True:
    bicycle_vel = p.readUserDebugParameter(env.bicycle_vel_param, physicsClientId=env.client)
    handlebar_angle = p.readUserDebugParameter(env.handlebar_angle_param, physicsClientId=env.client)

    roll_angle_control = roll_angle_pid(obs['obs'][0])
    action = np.array([handlebar_angle, -0.5, -roll_angle_control], np.float32)
    obs, _, terminated, truncated, infos = env.step(action)
    if terminated or truncated:
        obs, _ = env.reset()
        roll_angle_pid.reset()

env.close()
