import os
import time

import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from bicycle_dengh.resources.bicycle import Bicycle
from bicycle_dengh.resources.goal import Goal
import math
from utils import my_tools
from stable_baselines3.common.env_checker import check_env
from utils.normalize_action import NormalizeAction
from stable_baselines3 import PPO


class BicycleCameraEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, gui=False):
        self.goal = (0, 0)
        self.terminated = False
        self.truncated = False
        self.prev_dist_to_goal = 0.0
        self.gui = gui
        self.max_flywheel_vel = 120.0
        self.prev_goal_id = None
        self.camera_offset = [0.5, 0, 0.5]  # 相机相对于机器人的偏移 [前后, 左右, 上下]
        self.camera_up_vector = [0, 0, 1]  # 相机的上方向（z轴朝上）

        # action_space[车把角度，前后轮速度, 飞轮速度]
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1.57, 0.0, -self.max_flywheel_vel]),
            high=np.array([1.57, 5.0, self.max_flywheel_vel]),
            shape=(3,),
            dtype=np.float32)

        # 机器人与目标点距离, 机器人与目标点的角度, 翻滚角, 翻滚角角速度, 车把角度, 车把角速度, 后轮速度, 飞轮速度
        self.actual_observation_space = gym.spaces.box.Box(
            low=np.array([0.0, -math.pi, -math.pi, -15.0, -1.57, -15.0, 0.0, -self.max_flywheel_vel]),
            high=np.array([100.0, math.pi, math.pi, 15.0, 1.57, 15.0, 10.0, self.max_flywheel_vel]),
            shape=(8,),
            dtype=np.float32)

        self.observation_space = gym.spaces.box.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            shape=(8,),
            dtype=np.float32)

        # self.observation_space = gym.spaces.Dict({
        #     "image": gym.spaces.box.Box(low=0, high=255, shape=(36, 36, 1), dtype=np.uint8),
        #     "vector": gym.spaces.box.Box(
        #         low=np.array([0.0, -math.pi, -math.pi, -15.0, -1.57, -15.0, 0.0, -200]),
        #         high=np.array([100.0, math.pi, math.pi, 15.0, 1.57, 15.0, 10.0, 200]),
        #         shape=(8,),
        #         dtype=np.float32
        #     ),
        # })

        if gui:
            self.client = p.connect(p.GUI)
            self.camera_distance_param = p.addUserDebugParameter('camera_distance_param', 2, 60, 2)
            self.camera_yaw_param = p.addUserDebugParameter('camera_yaw_param', -180, 180, 0)
            self.camera_pitch_param = p.addUserDebugParameter('camera_pitch_param', -90, 90, -25)
        else:
            self.client = p.connect(p.DIRECT)

        p.setTimeStep(1. / 24., self.client)
        self.bicycle_vel_param = p.addUserDebugParameter('bicycle_vel_param', 0.0, 3.0, 1.0)
        self.handlebar_angle_param = p.addUserDebugParameter('handlebar_angle_param', -1.57, 1.57, 0)
        self.flywheel_param = p.addUserDebugParameter('flywheel_param', -40, 40, 0)

        self.bicycle = Bicycle(client=self.client, max_flywheel_vel=self.max_flywheel_vel)
        p.setGravity(0, 0, -10, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self.client)

    def step(self, action):
        self.bicycle.apply_action(action)
        p.stepSimulation(physicsClientId=self.client)
        obs = self.bicycle.get_observation()

        # 机器人位置与目标位置差x, 机器人位置与目标位置差y, 偏航角, 翻滚角, 翻滚角角速度, 车把角度, 车把角速度, 后轮速度, 飞轮速度
        distance_to_goal = math.sqrt((self.goal[0] - obs[0]) ** 2 + (self.goal[1] - obs[1]) ** 2)
        self.prev_dist_to_goal = distance_to_goal
        angle_to_target = my_tools.calculate_angle_to_target(obs[0], obs[1], obs[2], self.goal[0], self.goal[1])
        obs = [distance_to_goal, angle_to_target, obs[3], obs[4], obs[5], obs[6], obs[7], obs[8]]
        normalized_obs = my_tools.normalize_array_to_minus_one_to_one(obs, self.actual_observation_space.low,
                                                             self.actual_observation_space.high)

        if self.gui:
            bike_pos, _ = p.getBasePositionAndOrientation(self.bicycle.bicycleId, physicsClientId=self.client)
            camera_distance = p.readUserDebugParameter(self.camera_distance_param)
            camera_yaw = p.readUserDebugParameter(self.camera_yaw_param)
            camera_pitch = p.readUserDebugParameter(self.camera_pitch_param)
            p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, bike_pos)

        # 计算奖励值
        reward = self._reward_fun(obs, action)

        return normalized_obs, reward, self.terminated, self.truncated, {}

    def reset(self, seed=None, options=None):
        self.terminated = False
        self.truncated = False

        self.goal = my_tools.generate_goal_point()
        goal = Goal(self.client, self.goal)
        # 因为没有重置环境，每次reset后要清除先前的Goal
        if self.prev_goal_id is not None:
            p.removeBody(self.prev_goal_id)
        self.prev_goal_id = goal.id

        # 机器人位置与目标位置差x, 机器人位置与目标位置差y, 偏航角, 翻滚角, 翻滚角角速度, 车把角度, 车把角速度, 后轮速度, 飞轮速度
        obs = self.bicycle.reset()
        distance_to_goal = math.sqrt((self.goal[0] - obs[0]) ** 2 + (self.goal[1] - obs[1]) ** 2)
        self.prev_dist_to_goal = distance_to_goal
        angle_to_target = my_tools.calculate_angle_to_target(obs[0], obs[1], obs[2], self.goal[0], self.goal[1])
        obs = [distance_to_goal, angle_to_target, obs[3], obs[4], obs[5], obs[6], obs[7], obs[8]]
        normalized_obs = my_tools.normalize_array_to_minus_one_to_one(obs, self.actual_observation_space.low,
                                                             self.actual_observation_space.high)

        return normalized_obs, {}

    def _reward_fun(self, obs, action):
        self.terminated = False
        self.truncated = False

        # action [车把角度，前后轮速度, 飞轮速度]
        # obs [机器人与目标点距离, 机器人与目标点的角度, 翻滚角, 翻滚角角速度, 车把角度, 车把角速度, 后轮速度, 飞轮速度]
        roll_angle = obs[2]
        roll_angle_vel = obs[3]
        handlebar_angle_vel = obs[5]
        bicycle_vel = obs[6]
        flywheel_vel = obs[7]

        roll_angle_rwd = 0.4 * (0.3 - min(10.0 * (roll_angle ** 2), 0.3)) / 0.3
        roll_angle_vel_rwd = 0.3 * (225.0 - min((roll_angle_vel ** 2), 225.0)) / 225.0
        flywheel_rwd = 0.3 * (40.0 - min(0.001 * (flywheel_vel ** 2), 40.0)) / 40.0

        balance_rwd = 0.0
        if math.fabs(roll_angle) >= 0.17:
            self.terminated = True
            balance_rwd = -8.0
        elif 0.08 <= math.fabs(roll_angle) < 0.17:
            balance_rwd = -0.4
        elif math.fabs(roll_angle) <= 0.02:
            balance_rwd = 0.2

        #  到达目标点奖励
        goal_rwd = 0.0
        if math.fabs(obs[0]) <= 1.2:
            self.truncated = True
            goal_rwd = 10.0

        # 静止惩罚
        still_penalty = 0.0
        if math.fabs(bicycle_vel) <= 0.2:
            still_penalty = -1.0

        # 距离目标点奖励
        diff_dist_to_goal = self.prev_dist_to_goal - obs[0]
        distance_rwd = diff_dist_to_goal / (5.0 / 24.0)
        if diff_dist_to_goal > 0.0:
            distance_rwd = (1.0 / 10.0) * distance_rwd
        else:
            distance_rwd = (1.2 / 10.0) * distance_rwd

        total_reward = goal_rwd + distance_rwd + balance_rwd + still_penalty

        return total_reward

    def render(self):
        pass

    def close(self):
        p.disconnect(self.client)


if __name__ == '__main__':
    env = BicycleCameraEnv(gui=True)
    env = NormalizeAction(env)

    # It will check your custom environment and output additional warnings if needed
    # check_env(env, warn=True)

    models_dir = "D:\\data\\1-L\\9-bicycle\\bicycle-rl\\BicycleDengh\\output\\ppo_model_omni_0607_1820"
    model = PPO.load(models_dir)
    obs, _ = env.reset()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
        # time.sleep(1. / 24.)

