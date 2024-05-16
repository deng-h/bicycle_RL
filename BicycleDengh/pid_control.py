import pybullet as p
import time
from bicycle_dengh.resources import bicycle
from simple_pid import PID
import math
import gymnasium as gym

roll_angle_pid = PID(1100, 0, 0, setpoint=math.pi / 2)

try:
    # env = gym.make('BicycleDengh-v0', gui=True)
    env = gym.make('BalanceBicycleDengh-v0', gui=True)
    obs, _ = env.reset()
    while True:
        bicycle_vel_param = p.readUserDebugParameter(env.unwrapped.bicycle_vel_param)
        handlebar_angle_param = p.readUserDebugParameter(env.unwrapped.handlebar_angle_param)
        flywheel_param = p.readUserDebugParameter(env.unwrapped.flywheel_param)
        roll_angle_control = roll_angle_pid(obs[0])
        action = [-roll_angle_control]
        print(action[0])
        # action = [0, 0, 0]
        obs, reward, terminated, truncated, info = env.step(action)
        # if terminated or truncated:
        #     obs, _ = env.reset()

        time.sleep(1. / 24.)
finally:
    p.disconnect()
