"""
常用工具
"""
import math
import os
import random
import numpy as np
import pybullet as p
import json
from stable_baselines3.common.utils import set_random_seed
from typing import Callable
import gymnasium as gym

boundary_urdf = os.path.join(os.path.dirname(__file__), "../bicycle_dengh/resources/maze/maze_boundary.xml")
maze_config = os.path.join(os.path.dirname(__file__), "../bicycle_dengh/resources/maze/maze_config.json")
wall_length = 2.0
wall_height = 1.5
maze_size = 25
obstacle_coords = set()  # 存储障碍物的坐标


def degrees_to_radians(degrees):
    """
    将角度从度数转换为弧度
    :param degrees: 角度
    :return: 弧度
    """
    return degrees * (math.pi / 180.0)


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
    res = m * arr + c
    return np.array(res, dtype=np.float32)


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


def generate_goal():
    """
    随机生成目标点的坐标，不与障碍物坐标重合
    """
    while True:
        x = random.randint(-25, 25)
        y = random.randint(-3, 47)

        # 检查生成的坐标是否不在障碍物集合中
        if (x, y) not in obstacle_coords:
            return x - 0.5, y - 0.5


def build_maze(client):
    # Load maze boundary
    wall_id = p.loadURDF(boundary_urdf,
                         basePosition=[0.0, -3.0, 1.0],
                         baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
                         useFixedBase=True,
                         physicsClientId=client)

    with open(maze_config, 'r') as file:
        data = json.load(file)
        maze_layout = np.array(data['maze'])

    # 创建碰撞体和视觉体
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 1.0])
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 1.0],
                                       rgbaColor=[0.92, 0.94, 0.94, 1])

    obstacle_ids = [wall_id]
    # 遍历迷宫布局
    for y, row in enumerate(maze_layout):
        flipped_y = len(maze_layout) - y - 1  # 翻转y坐标
        for x, cell in enumerate(row):
            if cell == 'X':  # 如果该位置为障碍物
                obstacle_id = p.createMultiBody(baseCollisionShapeIndex=collision_shape,
                                                baseVisualShapeIndex=visual_shape,
                                                # 这里-25，-3是因为maze向Y轴负方向移动了3个单位，向X轴负方向移动了25个单位
                                                # +0.5是因为要让BOX刚好落在网格内，“不压线”
                                                basePosition=[x - 24.5, flipped_y - 2.5, 1.0],
                                                physicsClientId=client)
                obstacle_coords.add((x, flipped_y))  # 将障碍物的坐标添加到集合中
                obstacle_ids.append(obstacle_id)
            elif cell == 'S':  # 不让障碍物生成的区域
                obstacle_coords.add((x, flipped_y))  # 将障碍物的坐标添加到集合中
    return obstacle_ids


def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID 环境的ID，也就是环境的名称，如CartPole-v1
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = gym.make(env_id, gui=False)
        # 环境内部对action_space做了归一化，所以这里不需要再做归一化了
        # min_action = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        # max_action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        # env = RescaleAction(env, min_action=min_action, max_action=max_action)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init
