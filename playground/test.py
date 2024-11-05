import time
import numpy as np
import gymnasium as gym
import bicycle_dengh


roll_angles = []
steps = []
step = 0

env = gym.make('BicycleMaze-v0', gui=True)
obs, infos = env.reset()
for i in range(4000):
    action = np.array([0.0, -1.0, 1.0], np.float32)
    _, _, terminated, truncated, infos = env.step(action)

    # if terminated or truncated:
    #     obs, _ = env.reset()
    time.sleep(1. / 24.)


env.close()
