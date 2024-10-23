import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from bicycle_dengh.resources.bicycle import Bicycle
from bicycle_dengh.resources.goal import Goal
import math
import random


def generate_goal_point():
    # 随机生成距离，范围在10到20米之间
    distance = random.uniform(10, 20)
    
    # 随机生成角度，范围在0到180度之间（对应一二象限）
    angle_deg = random.uniform(0, 180)
    
    # 将角度转换为弧度
    angle_rad = math.radians(angle_deg)
    
    # 计算笛卡尔坐标
    x = distance * math.cos(angle_rad)
    y = distance * math.sin(angle_rad)
    
    return (x, y)

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


def calculate_angle_to_target(a, b, phi, x, y):
    """
    计算机器人与目标点之间的角度

    参数：
    a, b - 机器人的当前坐标 (a, b)
    phi - 机器人的当前偏航角，单位为弧度
    x, y - 目标点的坐标 (x, y)

    返回：
    机器人与目标点之间的角度，单位为弧度
    """
    # 计算目标点相对于机器人的方向
    delta_x = x - a
    delta_y = y - b

    # 计算目标方向的角度
    target_angle = math.atan2(delta_y, delta_x)

    # 计算机器人与目标点之间的相对角度
    angle_to_target = target_angle - phi

    # 将角度规范化到 [-π, π] 范围内
    angle_to_target = (angle_to_target + math.pi) % (2 * math.pi) - math.pi

    return angle_to_target


class BicycleDenghEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, gui=False):
        self.goal = (0, 0)
        self.terminated = False
        self.truncated = False
        self.prev_dist_to_goal = 0.0
        self.gui = gui
        self.max_flywheel_vel = 120.0
        self.prev_goal_id = None

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
        # 设置飞轮速度上限
        p.changeDynamics(self.bicycle.bicycleId,
                         self.bicycle.fly_wheel_joint,
                         maxJointVelocity=self.max_flywheel_vel,
                         physicsClientId=self.client)

    def step(self, action):
        self.bicycle.apply_action(action)
        p.stepSimulation(physicsClientId=self.client)
        obs = self.bicycle.get_observation()

        # 机器人位置与目标位置差x, 机器人位置与目标位置差y, 偏航角, 翻滚角, 翻滚角角速度, 车把角度, 车把角速度, 后轮速度, 飞轮速度
        distance_to_goal = math.sqrt((self.goal[0] - obs[0]) ** 2 + (self.goal[1] - obs[1]) ** 2)
        angle_to_target = calculate_angle_to_target(obs[0], obs[1], obs[2], self.goal[0], self.goal[1])
        obs = [distance_to_goal, angle_to_target, obs[3], obs[4], obs[5], obs[6], obs[7], obs[8]]
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
        self.prev_dist_to_goal = distance_to_goal

        return normalized_obs, reward, self.terminated, self.truncated, {"origin_obs": obs}

    def reset(self, seed=None, options=None):
        self.terminated = False
        self.truncated = False
        self.goal = generate_goal_point()
        goal = Goal(self.client, self.goal)
        # 因为没有重置环境，每次reset后要清除先前的Goal
        if self.prev_goal_id is not None:
            p.removeBody(self.prev_goal_id)
        self.prev_goal_id = goal.id

        # 机器人位置与目标位置差x, 机器人位置与目标位置差y, 偏航角, 翻滚角, 翻滚角角速度, 车把角度, 车把角速度, 后轮速度, 飞轮速度
        obs = self.bicycle.reset()
        distance_to_goal = math.sqrt((self.goal[0] - obs[0]) ** 2 + (self.goal[1] - obs[1]) ** 2)
        self.prev_dist_to_goal = distance_to_goal
        angle_to_target = calculate_angle_to_target(obs[0], obs[1], obs[2], self.goal[0], self.goal[1])
        obs = [distance_to_goal, angle_to_target, obs[3], obs[4], obs[5], obs[6], obs[7], obs[8]]

        normalized_obs = normalize_array_to_minus_one_to_one(obs, self.actual_observation_space.low,
                                                             self.actual_observation_space.high)
        normalized_obs = np.array(normalized_obs, dtype=np.float32)

        return normalized_obs, {"origin_obs": obs}

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
        # handlebar_angle_vel_rwd = 0.0
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
    env = BicycleDenghEnv(gui=False)