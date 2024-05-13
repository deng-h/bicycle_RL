import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from bicycle_dengh.resources.bicycle import Bicycle
from bicycle_dengh.resources.goal import Goal
from bicycle_dengh.resources.wall import Wall
import math
import random


class BicycleDenghEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, gui=False):
        # action_space[车把角度，前后轮速度，飞轮]
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1.57, 0, -40]),
            high=np.array([1.57, 5, 40]),
            shape=(3,),
            dtype=np.float32)

        # observation_space[离目标点的距离, 翻滚角roll, 车速, 车把角度]
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-100, -3.14, 0, -1.57]),
            high=np.array([100, 3.14, 5, 1.57]),
            shape=(4,),
            dtype=np.float32)

        self.bicycle = None
        self.goal = (0, 0)
        self.terminated = False
        self.truncated = False
        self.prev_dist_to_goal = 0.0
        self.prev_action = [0.0, 0.0, 0.0]
        self.gui = gui

        if gui:
            self.client = p.connect(p.GUI)
            self.bicycle_vel_param = p.addUserDebugParameter('bicycle_vel_param', 0.0, 3.0, 1.0)
            self.handlebar_angle_param = p.addUserDebugParameter('handlebar_angle_param', -1.57, 1.57, 0)
            self.flywheel_param = p.addUserDebugParameter('flywheel_param', -40, 40, 0)
            self.camera_distance_param = p.addUserDebugParameter('camera_distance_param', 2, 60, 2)
            self.camera_yaw_param = p.addUserDebugParameter('camera_yaw_param', -180, 180, 0)
            self.camera_pitch_param = p.addUserDebugParameter('camera_pitch_param', -90, 90, -25)
        else:
            self.client = p.connect(p.DIRECT)

        p.setTimeStep(1. / 24., self.client)

    def step(self, action):
        self.bicycle.apply_action(action)
        p.stepSimulation(physicsClientId=self.client)
        obs = self.bicycle.get_observation()

        dist_to_goal = math.sqrt(((obs[0] - self.goal[0]) ** 2 + (obs[1] - self.goal[1]) ** 2))
        self.prev_dist_to_goal = dist_to_goal

        if obs[0] >= 100 or obs[0] <= -100 or obs[1] >= 100 or obs[1] <= -100:
            self.terminated = True

        if self.gui:
            bike_pos, _ = p.getBasePositionAndOrientation(self.bicycle.bicycleId, physicsClientId=self.client)
            camera_distance = p.readUserDebugParameter(self.camera_distance_param)
            camera_yaw = p.readUserDebugParameter(self.camera_yaw_param)
            camera_pitch = p.readUserDebugParameter(self.camera_pitch_param)
            p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, bike_pos)

        # 计算奖励值
        # self._reward_fun(roll_angle=obs[2])
        roll_angle = obs[2]
        reward = self._reward_fun(roll_angle, action, self.prev_action)
        # print(np.array(self.prev_action) - np.array(action))
        self.prev_action = action

        # [离目标点的距离, 翻滚角roll, 车把角度]
        obs = np.array([dist_to_goal, obs[2], obs[3], obs[4]], dtype=np.float32)

        return obs, reward, self.terminated, self.truncated, {}

    def reset(self, seed=None, options=None):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -10, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self.client)

        self.terminated = False
        self.truncated = False

        self.bicycle = Bicycle(client=self.client)
        # 设置目标点
        x = (random.uniform(30, 40) if random.choice([True, False]) else random.uniform(-30, -40))
        y = (random.uniform(30, 40) if random.choice([True, False]) else random.uniform(-30, -40))
        self.goal = (x, y)
        Goal(self.client, self.goal)

        # Wall(self.client, [0, 0.7, 0])
        # Wall(self.client, [0, -0.7, 0])

        obs = self.bicycle.get_observation()
        self.prev_dist_to_goal = math.sqrt(((obs[0] - self.goal[0]) ** 2 + (obs[1] - self.goal[1]) ** 2))
        return np.array([self.prev_dist_to_goal, obs[2], obs[3], obs[4]], dtype=np.float32), {}

    def _reward_fun(self, roll_angle, action, prev_action):
        self.terminated = False
        self.truncated = False

        wheel_vel_change_reward = 0.0

        diff_handlebar_angle = math.fabs(action[0] - prev_action[0])
        diff_wheel_vel = math.fabs(action[1] - prev_action[1])
        diff_flywheel = math.fabs(action[2] - prev_action[2])

        # 车把角度变化不能太大
        if diff_handlebar_angle > 0.5:
            handlebar_angle_change_reward = -1.0
        else:
            handlebar_angle_change_reward = 1.0

        # 车速变化不能太大
        # if diff_wheel_vel > 0.5:
        #     wheel_vel_change_reward = -1.0
        # else:
        #     wheel_vel_change_reward = 1.0

        if roll_angle >= math.pi / 2 + 0.1 or roll_angle <= math.pi / 2 - 0.1:
            self.terminated = True
            balance_reward = -1.0
        else:
            balance_reward = 1.0
        
        total_reward = balance_reward + handlebar_angle_change_reward + wheel_vel_change_reward

        return total_reward

    def render(self):
        pass

    def close(self):
        p.disconnect(self.client)
