import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from bicycle_dengh.resources.balance_bicycle import BalanceBicycle
import math


class BalanceBicycleDenghEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, gui=False):
        self.bicycle = None
        self.terminated = False
        self.truncated = False
        self.gui = gui
        # 平衡奖励函数参数
        self.balance_alpha = 2.0
        self.balance_beta = 0.001
        self.balance_gamma = 0.0001
        self.balance_roll_angle_epsilon = 0.2
        self.balance_roll_angle_vel_epsilon = 3.0
        self.balance_delta = 10.0
        self.roll_angle_epsilon = 0.2

        self.max_flywheel_vel = 200.0
        self.max_torque = 200.0

        # [飞轮]
        self.action_space = gym.spaces.box.Box(
            low=np.array([-self.max_flywheel_vel]),
            high=np.array([self.max_flywheel_vel]),
            shape=(1,),
            dtype=np.float32)

        # [翻滚角, 翻滚角角速度, 飞轮角度, 飞轮角速度]
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-math.pi, -10.0, 0, -self.max_flywheel_vel]),
            high=np.array([math.pi, 10.0, 2 * math.pi, self.max_flywheel_vel]),
            shape=(4,),
            dtype=np.float32)

        if gui:
            self.client = p.connect(p.GUI)
            self.camera_distance_param = p.addUserDebugParameter('camera_distance_param', 1, 60, 2)
            self.camera_yaw_param = p.addUserDebugParameter('camera_yaw_param', -180, 180, 0)
            self.camera_pitch_param = p.addUserDebugParameter('camera_pitch_param', -90, 90, -25)
        else:
            self.client = p.connect(p.DIRECT)

        p.setTimeStep(1. / 24., self.client)
        self.bicycle_vel_param = p.addUserDebugParameter('bicycle_vel_param', 0.0, 3.0, 1.0)
        self.handlebar_angle_param = p.addUserDebugParameter('handlebar_angle_param', -1.57, 1.57, 0)
        self.flywheel_param = p.addUserDebugParameter('flywheel_param', -40, 40, 0)

    def step(self, action):
        self.bicycle.apply_action(action)
        p.stepSimulation(physicsClientId=self.client)
        obs = self.bicycle.get_observation()
        reward = self._reward_fun(obs, action)
        obs = np.array(obs, dtype=np.float32)

        if self.gui:
            bike_pos, _ = p.getBasePositionAndOrientation(self.bicycle.bicycleId, physicsClientId=self.client)
            camera_distance = p.readUserDebugParameter(self.camera_distance_param)
            camera_yaw = p.readUserDebugParameter(self.camera_yaw_param)
            camera_pitch = p.readUserDebugParameter(self.camera_pitch_param)
            p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, bike_pos)

        return obs, reward, self.terminated, self.truncated, {}

    def reset(self, seed=None, options=None):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -10, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self.client)

        self.terminated = False
        self.truncated = False

        self.bicycle = BalanceBicycle(client=self.client)
        # 设置飞轮速度上限
        p.changeDynamics(self.bicycle.bicycleId,
                         self.bicycle.fly_wheel_joint,
                         maxJointVelocity=self.max_flywheel_vel,
                         physicsClientId=self.client)

        obs = self.bicycle.get_observation()
        return np.array(obs, dtype=np.float32), {}

    def _reward_fun(self, obs, action):
        self.terminated = False
        self.truncated = False
        # 横滚角
        roll_angle = obs[0]
        # 横滚角速度
        roll_angle_vel = obs[1]
        # 惯性轮角速度
        flywheel_joint_vel = action[0]

        # 基本惩罚项
        reward = -(self.balance_alpha * (roll_angle - math.pi / 2) ** 2 +
                   self.balance_beta * roll_angle_vel ** 2 +
                   self.balance_gamma * flywheel_joint_vel ** 2)

        if math.fabs(roll_angle - math.pi / 2) <= self.roll_angle_epsilon:
            balance_reward = 2.0
        else:
            balance_reward = -2.0
            self.terminated = True

        # 额外奖励项
        # if (math.fabs(roll_angle - math.pi / 2) < self.balance_roll_angle_epsilon and
        #         math.fabs(roll_angle_vel) < self.balance_roll_angle_vel_epsilon):
        #     reward += self.balance_delta

        total_reward = reward + balance_reward

        return total_reward

    def render(self):
        pass

    def close(self):
        p.disconnect(self.client)
