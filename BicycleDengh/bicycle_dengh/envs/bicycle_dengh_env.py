import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from bicycle_dengh.resources.bicycle import Bicycle
from bicycle_dengh.resources.goal import Goal
import math
import random
from simple_pid import PID


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


class BicycleDenghEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, gui=False):
        self.goal = (0, 0)
        self.terminated = False
        self.truncated = False
        self.prev_dist_to_goal = 0.0
        self.gui = gui
        self.max_flywheel_vel = 200.0

        self.balance_alpha = 10.0
        self.balance_beta = 0.01

        self.roll_angle_pid = PID(2500, 100, 100, setpoint=0.0)

        # action_space[车把角度，前后轮速度, 飞轮速度]
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1.57, 0.0]),
            high=np.array([1.57, 5.0]),
            shape=(2,),
            dtype=np.float32)

        # 机器人位置与目标位置差x, 机器人位置与目标位置差y, 速度, 车把角度, 翻滚角, 翻滚角角速度, 飞轮角度, 飞轮角速度
        self.actual_observation_space = gym.spaces.box.Box(
            low=np.array([-50.0, -50.0, 0.0, -1.57]),
            high=np.array([50.0, 50.0, 5.0, 1.57]),
            # low=np.array([-50.0, -50.0, 0.0, -1.57, -math.pi, -15.0, 0.0, -self.max_flywheel_vel]),
            # high=np.array([50.0, 50.0, 5.0, 1.57, math.pi, 15.0, 2 * math.pi, self.max_flywheel_vel]),
            shape=(4,),
            dtype=np.float32)

        self.observation_space = gym.spaces.box.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0]),
            # low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            # high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            shape=(4,),
            dtype=np.float32)

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

        self.bicycle = Bicycle(client=self.client)
        p.setGravity(0, 0, -10, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self.client)

        # 设置目标点
        x = (random.uniform(10, 20) if random.choice([True, False]) else random.uniform(-20, -10))
        y = (random.uniform(10, 20) if random.choice([True, False]) else random.uniform(-20, -10))
        self.goal = (x, y)
        Goal(self.client, self.goal)

    def step(self, action):
        bicycle_obs = self.bicycle.get_observation()
        roll_angle_action = self.roll_angle_pid(bicycle_obs[2])
        all_action = [action[0], action[1], roll_angle_action]
        self.bicycle.apply_action(all_action)
        p.stepSimulation(physicsClientId=self.client)
        bicycle_obs = self.bicycle.get_observation()

        obs = [self.goal[0] - bicycle_obs[0], self.goal[1] - bicycle_obs[1],
               bicycle_obs[6], bicycle_obs[4]]

        # obs = [self.goal[0] - bicycle_obs[0], self.goal[1] - bicycle_obs[1],
        #        bicycle_obs[6], bicycle_obs[4],
        #        bicycle_obs[2], bicycle_obs[3],
        #        bicycle_obs[7], bicycle_obs[8]]

        normalized_obs = normalize_array_to_minus_one_to_one(obs, self.actual_observation_space.low,
                                                             self.actual_observation_space.high)
        normalized_obs = np.array(normalized_obs, dtype=np.float32)

        if self.gui:
            bike_pos, _ = p.getBasePositionAndOrientation(self.bicycle.bicycleId, physicsClientId=self.client)
            camera_distance = p.readUserDebugParameter(self.camera_distance_param)
            camera_yaw = p.readUserDebugParameter(self.camera_yaw_param)
            camera_pitch = p.readUserDebugParameter(self.camera_pitch_param)
            p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, bike_pos)

        # 计算奖励值
        reward = self._reward_fun(obs, action)

        return normalized_obs, reward, self.terminated, self.truncated, {"origin_obs": obs}

    def reset(self, seed=None, options=None):
        self.terminated = False
        self.truncated = False
        # 设置飞轮速度上限
        p.changeDynamics(self.bicycle.bicycleId,
                         self.bicycle.fly_wheel_joint,
                         maxJointVelocity=self.max_flywheel_vel,
                         physicsClientId=self.client)

        # 机器人位置与目标位置差x, 机器人位置与目标位置差y, 速度, 车把角度, 翻滚角, 翻滚角角速度, 飞轮角度, 飞轮角速度
        bicycle_obs = self.bicycle.reset()

        self.roll_angle_pid.reset()

        obs = [self.goal[0] - bicycle_obs[0], self.goal[1] - bicycle_obs[1],
               bicycle_obs[6], bicycle_obs[4]]

        # obs = [self.goal[0] - bicycle_obs[0], self.goal[1] - bicycle_obs[1],
        #        bicycle_obs[6], bicycle_obs[4],
        #        bicycle_obs[2], bicycle_obs[3],
        #        bicycle_obs[7], bicycle_obs[8]]
        normalized_obs = normalize_array_to_minus_one_to_one(obs, self.actual_observation_space.low,
                                                             self.actual_observation_space.high)
        normalized_obs = np.array(normalized_obs, dtype=np.float32)

        return normalized_obs, {"origin_obs": obs}

    def _reward_fun(self, obs, action):
        self.terminated = False
        self.truncated = False

        # 横滚角
        # roll_angle = obs[4]
        # 横滚角速度
        # roll_angle_vel = obs[5]
        # 飞轮速度
        # flywheel_vel = action[2]
        #
        # reward_roll_angle = (0.3 - min(self.balance_alpha * (roll_angle ** 2), 0.3)) / 0.3
        # reward_roll_angle_vel = (144.0 - min(self.balance_beta * (roll_angle_vel ** 2), 144.0)) / 144.0
        # action_penalty = (40.0 - min(0.001 * (flywheel_vel ** 2), 40.0)) / 40.0
        #
        # reward_1 = 0.4 * reward_roll_angle + 0.3 * reward_roll_angle_vel + 0.3 * action_penalty

        # balance_reward = 0.0
        # if math.fabs(roll_angle) >= 0.17:
        #     self.terminated = True
        #     balance_reward = -10.0
        # elif math.fabs(roll_angle) <= 0.02:
        #     balance_reward = 2.0

        reward_distance_x = (2500 - min(obs[0] ** 2, 2500)) / 2500
        reward_distance_y = (2500 - min(obs[1] ** 2, 2500)) / 2500

        reward_2 = 0.5 * reward_distance_x + 0.5 * reward_distance_y

        total_reward = reward_2
        return total_reward

    def render(self):
        pass

    def close(self):
        p.disconnect(self.client)


if __name__ == '__main__':
    env = BicycleDenghEnv(gui=False)
