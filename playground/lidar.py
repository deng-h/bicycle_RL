import math
import pybullet as p
import time
import pybullet_data
import numpy as np
import matplotlib.pyplot as plt
import keyboard
from utils import my_tools


def map_to_image(ray_results, img_size=300, x_range=(-25, 25), y_range=(-4, 46)):
    """
    将 rayTestBatch 的 hit position 映射到二维图像上，并反转y轴方向。

    ray_results: rayTestBatch 返回的结果列表
    img_size: 图像的大小（例如128x128）
    x_range: X轴的范围
    y_range: Y轴的范围

    返回: 二维图像，表示环境中的障碍物
    """
    # 创建一个空的二维图像（初始化为全0），并设置数据类型为 uint8
    radar_image = np.zeros((img_size, img_size), dtype=np.uint8)

    # X 和 Y 轴的物理范围
    x_min, x_max = x_range
    y_min, y_max = y_range
    width = x_max - x_min
    height = y_max - y_min

    # 遍历所有的 rayTestBatch 结果
    for result in ray_results:
        hit_position = result[3]  # 获取 hit_position (x, y, z)
        x, y = hit_position[0], hit_position[1]

        # 将物理坐标映射到图像坐标
        u = int((x - x_min) / width * img_size)
        v = img_size - 1 - int((y - y_min) / height * img_size)

        # 确保坐标在图像范围内
        if 0 <= u < img_size and 0 <= v < img_size:
            radar_image[v, u] = 255  # 将障碍物位置标记为白色

    return radar_image


physicsClient = p.connect(p.GUI)  # or p.DIRECT for non-graphical version
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # optionally
p.setGravity(0, 0, -10)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)  # 关闭阴影效果，透明的陀螺仪会显示出来，问题不大
planeId = p.loadURDF("plane.urdf")
# obstacle_id2 = p.loadURDF("cube.urdf", [-6, 35, 0.5], globalScaling=0.5)
obstacle_id2 = p.loadURDF("D:\\data\\1-L\\9-bicycle\\bicycle-rl\\bicycle_dengh\\resources\\simplegoal.xml",
                          basePosition=[0, 0, 0.5])
obstacle_ids = my_tools.build_maze(physicsClient)
p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

lidar_origin_offset = [0, 0, 1.0]  # 激光雷达相对于小车的位置偏移量
numRays = 1080
rayLen = 100

x = p.addUserDebugParameter("x", -25, 25, -6)
y = p.addUserDebugParameter("y", -3, 47, 35)
camera_distance_param = p.addUserDebugParameter('camera_distance_param', 2, 60, 30)
camera_yaw_param = p.addUserDebugParameter('camera_yaw_param', -180, 180, 0)
camera_pitch_param = p.addUserDebugParameter('camera_pitch_param', -90, 90, -89)

while True:
    p.resetBasePositionAndOrientation(obstacle_id2, [p.readUserDebugParameter(x),
                                                     p.readUserDebugParameter(y), 0.5], [0, 0, 0, 1])
    obstacle_pos = p.getBasePositionAndOrientation(obstacle_id2)[0]

    camera_distance = p.readUserDebugParameter(camera_distance_param)
    camera_yaw = p.readUserDebugParameter(camera_yaw_param)
    camera_pitch = p.readUserDebugParameter(camera_pitch_param)
    p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, [0, 18, 0])

    rayFrom = []
    rayTo = []

    for i in range(numRays):
        rayFrom.append([obstacle_pos[0] + lidar_origin_offset[0], obstacle_pos[1] + lidar_origin_offset[1],
                        obstacle_pos[2] + lidar_origin_offset[2]])
        rayTo.append([
            rayFrom[i][0] + rayLen * math.sin(2. * math.pi * float(i) / numRays),
            rayFrom[i][1] + rayLen * math.cos(2. * math.pi * float(i) / numRays),
            rayFrom[i][2]
        ])

    # lidar_origin = [obstacle_pos[0] + lidar_origin_offset[0], obstacle_pos[1] + lidar_origin_offset[1],
    #                 obstacle_pos[2] + lidar_origin_offset[2]]
    # 使用 rayTestBatch 进行批量检测
    # results格式：线数x元组，元组格式(碰撞物体的id, 碰撞物体的link索引, 沿射线的命中率范围 [0,1], 碰撞点的世界坐标, 碰撞点归一化的世界坐标)
    results = p.rayTestBatch(rayFromPositions=rayFrom, rayToPositions=rayTo)
    # 解析检测结果，提取距离信息
    # distances = []
    # for i, result in enumerate(results):
    #     if result[0] < 0:
    #         # 未检测到碰撞，设为最大探测距离
    #         distance = lidar_range
    #         # end_position = [lidar_origin[0] + lidar_range * ray_directions[i][0],
    #         #                 lidar_origin[1] + lidar_range * ray_directions[i][1],
    #         #                 lidar_origin[2]]
    #         # p.addUserDebugLine(rayFrom[i], rayTo[i], lineColorRGB=[0, 1, 0], lineWidth=1.0)
    #     else:
    #         # 获取到碰撞点，计算激光雷达到碰撞点的距离
    #         hit_position = result[3]
    #         distance = np.linalg.norm(np.array(hit_position) - np.array(lidar_origin))
    #         # p.addUserDebugLine(rayFrom[i], hit_position, lineColorRGB=[1, 0, 0], lineWidth=1.0)
    #     distances.append(distance)

    if keyboard.is_pressed('q'):
        # 创建雷达图像
        radar_image = map_to_image(results)
        plt.imshow(radar_image, cmap='gray')
        plt.colorbar()
        plt.title("Radar Image")
        plt.show()

    p.stepSimulation()
    time.sleep(1. / 24.)
