import gymnasium
import numpy as np
import pybullet as p
import pybullet_data
from bicycle_dengh.resources.balance_bicycle import BalanceBicycle
import math
import time
import csv


class BicycleBalanceEnv(gymnasium.Env):
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, gui=False):
        self.terminated = False
        self.truncated = False
        self.gui = gui
        self.max_flywheel_vel = 120.
        self._max_episode_steps = 10000
        self._elapsed_steps = None
        self.roll_angle_data = []
        self.flywheel_vel = []

        self.action_space = gymnasium.spaces.box.Box(low=-1., high=1., shape=(1,), dtype=np.float32)

        # [飞轮]
        self.actual_action_space = gymnasium.spaces.box.Box(
            low=np.array([-self.max_flywheel_vel]),
            high=np.array([self.max_flywheel_vel]),
            shape=(1,),
            dtype=np.float32)

        # Retrieve the max/min values，环境内部对action_space做了归一化
        self.action_low, self.action_high = self.actual_action_space.low, self.actual_action_space.high

        # [翻滚角, 翻滚角角速度, 飞轮角度, 飞轮角速度]
        self.observation_space = gymnasium.spaces.box.Box(
            low=np.array([-math.pi, -15.0, 0.0, -self.max_flywheel_vel]),
            high=np.array([math.pi, 15.0, 2 * math.pi, self.max_flywheel_vel]),
            shape=(4,),
            dtype=np.float32)

        if self.gui:
            self.client = p.connect(p.GUI)
            self.camera_distance_param = p.addUserDebugParameter('camera_distance_param', 2, 60, 5)
            self.camera_yaw_param = p.addUserDebugParameter('camera_yaw_param', -180, 180, 0)
            self.camera_pitch_param = p.addUserDebugParameter('camera_pitch_param', -90, 90, -30)
            self.bicycle_vel_param = p.addUserDebugParameter('bicycle_vel_param', 0.0, 3.0, 1.0)
            self.handlebar_angle_param = p.addUserDebugParameter('handlebar_angle_param', -1.57, 1.57, 0)
            self.flywheel_param = p.addUserDebugParameter('flywheel_param', -40, 40, 0)
        else:
            self.client = p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)  # 关闭阴影效果，透明的陀螺仪会显示出来，问题不大

        self.bicycle = BalanceBicycle(self.client, self.max_flywheel_vel)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        friction_coefficient = 1.5  # 摩擦系数
        # 更改地面物体的动力学参数，包括摩擦系数
        p.changeDynamics(plane_id, -1, lateralFriction=friction_coefficient)  # -1表示所有部件
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.setGravity(0, 0, -10, physicsClientId=self.client)
        p.setTimeStep(1. / 50., self.client)

    def step(self, action):
        rescaled_action = self._rescale_action(action)  # Rescale action from [-1, 1] to original [low, high] interval
        self.bicycle.apply_action(rescaled_action)
        p.stepSimulation(physicsClientId=self.client)
        obs = self.bicycle.get_observation()
        reward = self._reward_fun(obs)  # 计算奖励值

        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            self.truncated = True

        if self.gui:
            bike_pos, _ = p.getBasePositionAndOrientation(self.bicycle.bicycleId, physicsClientId=self.client)
            camera_distance = p.readUserDebugParameter(self.camera_distance_param)
            camera_yaw = p.readUserDebugParameter(self.camera_yaw_param)
            camera_pitch = p.readUserDebugParameter(self.camera_pitch_param)
            p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, bike_pos)
            # self.roll_angle_data.append(obs[0])
            self.flywheel_vel.append(obs[3])

        return np.array(obs, dtype=np.float32), reward, self.terminated, self.truncated, {}

    def reset(self, seed=None, options=None):
        self.terminated = False
        self.truncated = False
        self._elapsed_steps = 0
        obs = self.bicycle.reset()
        return np.array(obs, dtype=np.float32), {}

    def _reward_fun(self, obs):
        self.terminated = False
        self.truncated = False

        roll_angle = obs[0]
        wheel_vel = obs[3]

        """平衡奖励"""
        # balance_rwd = 0.0
        # if math.fabs(roll_angle) >= 0.35:
        #     self.terminated = True
        #     balance_rwd = -10.0
        # else:
        #     # 计算奖励值，倾角越小，奖励越大
        #     balance_rwd = 1.0 - (math.fabs(roll_angle) / 0.35) * 2.0
        #     # 限制奖励值在范围[-max_reward, max_reward]之间
        #     balance_rwd = max(-1.0, min(1.0, balance_rwd))
        #
        # r2 = -0.001 * math.fabs(wheel_vel)  # 惯性轮速度惩罚

        balance_rwd_dui_bi = 0.0
        if math.fabs(roll_angle) >= 0.35:
            self.terminated = True
            balance_rwd_dui_bi = -10.0
        elif math.fabs(roll_angle) >= 0.25:
            balance_rwd_dui_bi = -1.0
        elif math.fabs(roll_angle) >= 0.15:
            balance_rwd_dui_bi = -0.5
        else:
            balance_rwd_dui_bi = 1.0

        # return balance_rwd + r2
        return balance_rwd_dui_bi

    def render(self):
        pass

    def close(self):
        p.disconnect(self.client)
        # 以写入模式打开 CSV 文件
        with open("离散奖励函数的飞轮数据.csv", mode='w', newline='') as file:
            # 创建一个 CSV 写入器对象
            writer = csv.writer(file)
            for value in self.flywheel_vel:
                writer.writerow([value])  # 每个值写成一行

    def _rescale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high] (no need for symmetric action space)
        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        return self.action_low + (0.5 * (scaled_action + 1.0) * (self.action_high - self.action_low))


if __name__ == '__main__':
    env = gymnasium.make('BicycleBalance-v0', gui=True)
    obs, _ = env.reset()
    for i in range(40000):
        action = np.array([0.5], np.float32)
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()
        # time.sleep(1)

    env.close()
