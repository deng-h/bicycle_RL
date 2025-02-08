import gymnasium
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
import math
from simple_pid import PID
import time

from bicycle_dengh.resources.small_bicycle import SmallBicycle


class BalanceEnvS(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, gui=False):
        self.terminated = False
        self.truncated = False
        self.gui = gui
        self.max_flywheel_vel = 120.0

        self.action_space = gymnasium.spaces.box.Box(low=-1., high=1., shape=(1,), dtype=np.float32)

        self.actual_action_space = gym.spaces.box.Box(
            low=np.array([-self.max_flywheel_vel]),
            high=np.array([self.max_flywheel_vel]),
            shape=(1,),
            dtype=np.float32)

        # Retrieve the max/min values，环境内部对action_space做了归一化
        self.action_low, self.action_high = self.actual_action_space.low, self.actual_action_space.high

        self.observation_space = gym.spaces.box.Box(
            low=np.array([-3.14]),
            high=np.array([3.14]),
            shape=(1,),
            dtype=np.float32)

        if gui:
            self.client = p.connect(p.GUI)
            self.camera_distance_param = p.addUserDebugParameter('camera_distance_param', 1, 60, 2)
            self.camera_yaw_param = p.addUserDebugParameter('camera_yaw_param', -180, 180, 0)
            self.camera_pitch_param = p.addUserDebugParameter('camera_pitch_param', -90, 90, -25)
            self.bicycle_vel_param = p.addUserDebugParameter('bicycle_vel_param', 0.0, 3.0, 1.0)
            self.handlebar_angle_param = p.addUserDebugParameter('handlebar_angle_param', -1.57, 1.57, 0)
            self.flywheel_param = p.addUserDebugParameter('flywheel_param', -40, 40, 0)
        else:
            self.client = p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)  # 关闭阴影效果，透明的陀螺仪会显示出来，问题不大

        self.bicycle = SmallBicycle(client=self.client, max_flywheel_vel=self.max_flywheel_vel)
        p.setGravity(0, 0, -10.0, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

    def step(self, action):
        # Rescale action from [-1, 1] to original [low, high] interval
        rescaled_action = self._rescale_action(action)
        self.bicycle.apply_action(rescaled_action)
        p.stepSimulation(physicsClientId=self.client)
        obs = self.bicycle.get_observation()
        reward = self._reward_fun(obs, action)

        if self.gui:
            bike_pos, _ = p.getBasePositionAndOrientation(self.bicycle.bicycleId, physicsClientId=self.client)
            camera_distance = p.readUserDebugParameter(self.camera_distance_param)
            camera_yaw = p.readUserDebugParameter(self.camera_yaw_param)
            camera_pitch = p.readUserDebugParameter(self.camera_pitch_param)
            p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, bike_pos)

        return np.array(obs, dtype=np.float32), reward, self.terminated, self.truncated, {}

    def reset(self, seed=None, options=None):
        self.terminated = False
        self.truncated = False

        obs = self.bicycle.reset()
        return np.array(obs, dtype=np.float32), {}

    def _reward_fun(self, obs, action):
        self.terminated = False
        self.truncated = False

        roll_angle = 1.57 - obs[0]

        """平衡奖励"""
        balance_rwd = 0.0
        if math.fabs(roll_angle) >= 0.35:
            self.terminated = True
            balance_rwd = -10.0
        else:
            # 计算奖励值，倾角越小，奖励越大
            balance_rwd = 1.0 - (math.fabs(roll_angle) / 0.35) * 2.0
            # 限制奖励值在范围[-max_reward, max_reward]之间
            balance_rwd = max(-1.0, min(1.0, balance_rwd))

        total_reward = balance_rwd

        return total_reward

    def render(self):
        pass

    def close(self):
        p.disconnect(self.client)

    def _rescale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high] (no need for symmetric action space)
        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        return self.action_low + (0.5 * (scaled_action + 1.0) * (self.action_high - self.action_low))


if __name__ == '__main__':
    roll_angle_pid = PID(50, 0, 0, setpoint=1.57)
    env = gym.make('BalanceEnvS-v0', gui=True)
    obs, _ = env.reset()

    while True:
        # bicycle_vel = p.readUserDebugParameter(env.bicycle_vel_param, physicsClientId=env.client)
        # handlebar_angle = p.readUserDebugParameter(env.handlebar_angle_param, physicsClientId=env.client)
        # flywheel_param = p.readUserDebugParameter(env.flywheel_param, physicsClientId=env.client)
        roll_angle_control = roll_angle_pid(obs[0])
        # action = np.array([handlebar_angle, -0.5, -roll_angle_control], np.float32)
        action = np.array([roll_angle_control], np.float32)
        # print(f"action: {action}, obs[0]: {obs[0]}")
        obs, _, terminated, truncated, infos = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
            roll_angle_pid.reset()
        # time.sleep(1. / 24.)
