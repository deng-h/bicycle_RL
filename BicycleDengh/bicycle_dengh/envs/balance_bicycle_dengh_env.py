import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from bicycle_dengh.resources.balance_bicycle import BalanceBicycle
import math
from bicycle_dengh.resources.wall import Wall


def normalize_array_to_minus_one_to_one(arr, a, b):
    """
    将数组arr从区间[a, b]归一化到[-1, 1]

    参数:
    arr -- 要归一化的数组
    a -- 区间下限
    b -- 区间上限

    返回:
    归一化后的数组
    """
    if a.all() == b.all():
        raise ValueError("a 和 b 不能相等")

    m = 2 / (b - a)
    c = - (b + a) / (b - a)
    return m * arr + c


class BalanceBicycleDenghEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, gui=False):
        self.terminated = False
        self.truncated = False
        self.gui = gui

        # 平衡奖励函数参数
        self.balance_alpha = 10.0
        self.balance_beta = 0.01
        self.balance_gamma = 0.0

        self.max_flywheel_vel = 120.0
        self.last_action = [0]

        # [飞轮]
        self.action_space = gym.spaces.box.Box(
            low=np.array([-self.max_flywheel_vel]),
            high=np.array([self.max_flywheel_vel]),
            shape=(1,),
            dtype=np.float32)

        # [翻滚角, 翻滚角角速度, 飞轮角度, 飞轮角速度]
        self.actual_observation_space = gym.spaces.box.Box(
            low=np.array([-math.pi, -15.0, 0.0, -self.max_flywheel_vel]),
            high=np.array([math.pi, 15.0, 2 * math.pi, self.max_flywheel_vel]),
            shape=(4,),
            dtype=np.float32)

        self.observation_space = gym.spaces.box.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            shape=(4,),
            dtype=np.float32)

        if gui:
            self.client = p.connect(p.GUI)
            self.camera_distance_param = p.addUserDebugParameter('camera_distance_param', 1, 60, 2)
            self.camera_yaw_param = p.addUserDebugParameter('camera_yaw_param', -180, 180, 0)
            self.camera_pitch_param = p.addUserDebugParameter('camera_pitch_param', -90, 90, -25)
        else:
            self.client = p.connect(p.DIRECT)

        self.bicycle = BalanceBicycle(client=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self.client)

        p.setTimeStep(1. / 24., self.client)
        self.bicycle_vel_param = p.addUserDebugParameter('bicycle_vel_param', 0.0, 3.0, 1.0)
        self.handlebar_angle_param = p.addUserDebugParameter('handlebar_angle_param', -1.57, 1.57, 0)
        self.flywheel_param = p.addUserDebugParameter('flywheel_param', -40, 40, 0)

    def step(self, action):
        self.bicycle.apply_action(action)
        p.stepSimulation(physicsClientId=self.client)
        obs = self.bicycle.get_observation()
        reward = self._reward_fun(obs, action)

        normalized_obs = normalize_array_to_minus_one_to_one(obs, self.actual_observation_space.low,
                                                             self.actual_observation_space.high)
        normalized_obs = np.array(normalized_obs, dtype=np.float32)

        if self.gui:
            bike_pos, _ = p.getBasePositionAndOrientation(self.bicycle.bicycleId, physicsClientId=self.client)
            camera_distance = p.readUserDebugParameter(self.camera_distance_param)
            camera_yaw = p.readUserDebugParameter(self.camera_yaw_param)
            camera_pitch = p.readUserDebugParameter(self.camera_pitch_param)
            p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, bike_pos)

        return normalized_obs, reward, self.terminated, self.truncated, {"origin_obs": obs}

    def reset(self, seed=None, options=None):
        self.terminated = False
        self.truncated = False
        self.last_action = [0]
        # 设置飞轮速度上限
        p.changeDynamics(self.bicycle.bicycleId,
                         self.bicycle.fly_wheel_joint,
                         maxJointVelocity=self.max_flywheel_vel,
                         physicsClientId=self.client)

        obs = self.bicycle.reset()
        normalized_obs = normalize_array_to_minus_one_to_one(obs, self.actual_observation_space.low,
                                                             self.actual_observation_space.high)
        return np.array(normalized_obs, dtype=np.float32), {"origin_obs": obs}

    def _reward_fun(self, obs, action):
        self.terminated = False
        self.truncated = False
        # 横滚角
        roll_angle = obs[0]
        # 横滚角速度
        roll_angle_vel = obs[1]
        # 惯性轮角速度
        # flywheel_joint_vel = action[0]
        reward_roll_angle = (0.3 - min(self.balance_alpha * (roll_angle ** 2), 0.3)) / 0.3
        reward_roll_angle_vel = (144.0 - min(self.balance_beta * (roll_angle_vel ** 2), 144.0)) / 144.0
        action_penalty = (72.0 - min(0.005 * (action[0] ** 2), 72.0)) / 72.0

        # print(f"reward_roll_angle: {self.balance_alpha * roll_angle ** 2:.2f}, "
        #       f"reward_roll_angle_vel: {self.balance_beta * roll_angle_vel ** 2:.2f}")

        reward = 0.4 * reward_roll_angle + 0.3 * reward_roll_angle_vel + 0.3 * action_penalty

        balance_reward = 0.0
        if math.fabs(roll_angle) >= 0.17:
            self.terminated = True
            balance_reward = -10.0
        elif math.fabs(roll_angle) <= 0.02:
            balance_reward = 10.0

        total_reward = reward + balance_reward

        return total_reward

    def render(self):
        pass

    def close(self):
        p.disconnect(self.client)

