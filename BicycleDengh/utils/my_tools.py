"""
常用工具
"""
import math
import random


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