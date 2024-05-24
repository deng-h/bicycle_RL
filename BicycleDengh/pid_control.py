import time
from bicycle_dengh.resources import bicycle
from simple_pid import PID
import math
import gymnasium as gym
import csv

roll_angles = []
steps = []
step = 0

roll_angle_pid = PID(2500, 100, 100, setpoint=0.0)  # 初始翻滚角为0.0°的情况
# roll_angle_pid = PID(1000, 150, 0, setpoint=0.0)  # 初始翻滚角为3.0°的情况
# roll_angle_pid = PID(1000, 200, 0, setpoint=0.0)  # 初始翻滚角为5.0°的情况
# roll_angle_pid = PID(900, 60, 0, setpoint=0.0)  # 初始翻滚角为7.0°的情况
# env = gym.make('BicycleDengh-v0', gui=True)
env = gym.make('BalanceBicycleDengh-v0', gui=True)
obs, infos = env.reset()
for i in range(4000):
    origin_obs = infos['origin_obs']
    roll_angle_control = roll_angle_pid(origin_obs[0])
    action = [-roll_angle_control]
    obs, _, terminated, truncated, infos = env.step(action)

    # print(f"roll_angle:{obs[0]:.2f}, flywheel_vel:{obs[3]:.2f}")
    # Record the roll_angle and step number
    # roll_angles.append(origin_obs[0])
    # steps.append(step)
    # print(f"roll_angle: {origin_obs[0]:.2f}, i={i}")
    # step += 1
    # if step % 1000 == 0:
    #     print(f"roll_angle: {origin_obs[0]:.2f}, i={i}")

    if terminated or truncated:
        obs, _ = env.reset()
        roll_angle_pid.reset()
        step = 0
    time.sleep(1. / 24.)


env.close()
# with open('roll_angle_data_7_degree.csv', 'w', newline='') as csvfile:
#     fieldnames = ['Step', 'Roll Angle']
#     writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
#
#     writer.writeheader()
#     for step, roll_angle in zip(steps, roll_angles):
#         writer.writerow({'Step': step, 'Roll Angle': roll_angle})
#
# print("Data saved to roll_angle_data.csv")
