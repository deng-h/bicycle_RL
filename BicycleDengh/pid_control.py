import pybullet as p
import time
from bicycle_dengh.resources import bicycle
from simple_pid import PID
import math
import gymnasium as gym

roll_angle_pid = PID(2500, 0, 100, setpoint=0.0)
try:
    # env = gym.make('BicycleDengh-v0', gui=True)
    env = gym.make('BalanceBicycleDengh-v0', gui=True)
    obs, _ = env.reset()
    while True:
        roll_angle_control = roll_angle_pid(obs[0])
        action = [-roll_angle_control]
        obs, _, terminated, truncated, _ = env.step(action)
        print(f"roll_angle:{obs[0]:.2f}, flywheel_vel:{obs[3]:.2f}")
        if terminated or truncated:
            obs, _ = env.reset()
            roll_angle_pid.reset()

        time.sleep(1. / 24.)
finally:
    p.disconnect()
