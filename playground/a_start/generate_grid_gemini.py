import pybullet as p
import pybullet_data
import numpy as np
import time
from a_star_algo import a_star_pathfinding
from create_grid_map import create_grid_map2
from visualize_path import visualize_path, smooth_path_bezier


def create_obstacle(client_id, obstacle_positions):
    """
    创建一个或多个障碍物。

    参数:
        client_id: PyBullet 客户端 ID
        obstacle_positions: 障碍物位置列表，每个位置是一个列表 [x, y, z]

    返回:
        障碍物的 bodyId 列表
    """
    collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 1.0])
    visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 1.0], rgbaColor=[0.92, 0.94, 0.94, 1])
    obstacle_ids = []
    for obstacle_pos in obstacle_positions:
        obstacle_id = p.createMultiBody(baseCollisionShapeIndex=collision_shape,
                                            baseVisualShapeIndex=visual_shape,
                                            basePosition=obstacle_pos,
                                            physicsClientId=client_id)
        obstacle_ids.append(obstacle_id)
    return obstacle_ids


if __name__ == '__main__':
    physicsClient = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    planeId = p.loadURDF("plane.urdf")

    # 设置俯视视角
    # 相机距离
    camera_distance = 10
    # 相机偏航角（水平旋转）
    camera_yaw = 0
    # 相机俯仰角（垂直旋转），-90 度表示俯视
    camera_pitch = -85
    # 相机目标位置
    camera_target_position = [15, 15, 10]

    # 重置调试可视化相机
    p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)

    # 定义不同的障碍物位置列表
    obstacle_positions_set1 = [
        [8 + 0.5, 5 + 0.5, 1],
        [8 + 0.5, 6 + 0.5, 1],
        [8 + 0.5, 7 + 0.5, 1],
        [8 + 0.5, 8 + 0.5, 1],
        [8 + 0.5, 9 + 0.5, 1],
        [8 + 0.5, 10 + 0.5, 1],
        [8 + 0.5, 11 + 0.5, 1],
        [8 + 0.5, 12 + 0.5, 1],
        [9 + 0.5, 12 + 0.5, 1],
        [10 + 0.5, 12 + 0.5, 1],
        [11 + 0.5, 12 + 0.5, 1],
        [12 + 0.5, 12 + 0.5, 1],
        [13 + 0.5, 12 + 0.5, 1],
        [14 + 0.5, 12 + 0.5, 1],
        [14 + 0.5, 13 + 0.5, 1],
        [14 + 0.5, 14 + 0.5, 1],
        [14 + 0.5, 15 + 0.5, 1],
        [14 + 0.5, 16 + 0.5, 1],
        [14 + 0.5, 17 + 0.5, 1],
        [14 + 0.5, 18 + 0.5, 1],
        [14 + 0.5, 19 + 0.5, 1],
        [14 + 0.5, 20 + 0.5, 1],
        [14 + 0.5, 21 + 0.5, 1],
        [14 + 0.5, 22 + 0.5, 1],
        [13 + 0.5, 22 + 0.5, 1],
        [12 + 0.5, 22 + 0.5, 1],
        [11 + 0.5, 22 + 0.5, 1],
        [10 + 0.5, 22 + 0.5, 1],
        [9 + 0.5, 22 + 0.5, 1],
        [8 + 0.5, 22 + 0.5, 1],
        [8 + 0.5, 23 + 0.5, 1],
        [8 + 0.5, 24 + 0.5, 1],
        [8 + 0.5, 25 + 0.5, 1],
        [8 + 0.5, 26 + 0.5, 1],
        [7 + 0.5, 26 + 0.5, 1],
        [6 + 0.5, 26 + 0.5, 1],
        [5 + 0.5, 26 + 0.5, 1],

        [0 + 0.5, 15 + 0.5, 1],
        [1 + 0.5, 15 + 0.5, 1],
        [2 + 0.5, 15 + 0.5, 1],
        [3 + 0.5, 15 + 0.5, 1],

        [16 + 0.5, 6 + 0.5, 1],
        [16 + 0.5, 7 + 0.5, 1],
        [17 + 0.5, 6 + 0.5, 1],
        [17 + 0.5, 7 + 0.5, 1],

        [25 + 0.5, 3 + 0.5, 1],
        [25 + 0.5, 4 + 0.5, 1],
        [26 + 0.5, 3 + 0.5, 1],
        [26 + 0.5, 4 + 0.5, 1],

        [18 + 0.5, 26 + 0.5, 1],
        [18 + 0.5, 27 + 0.5, 1],
        [19 + 0.5, 26 + 0.5, 1],
        [19 + 0.5, 27 + 0.5, 1],

        [23 + 0.5, 10 + 0.5, 1],
        [24 + 0.5, 10 + 0.5, 1],
        [25 + 0.5, 10 + 0.5, 1],
        [22 + 0.5, 10 + 0.5, 1],
        [22 + 0.5, 11 + 0.5, 1],
        [22 + 0.5, 12 + 0.5, 1],
        [22 + 0.5, 13 + 0.5, 1],
        [22 + 0.5, 14 + 0.5, 1],
        [22 + 0.5, 15 + 0.5, 1],
        [22 + 0.5, 16 + 0.5, 1],
        [22 + 0.5, 17 + 0.5, 1],
        [22 + 0.5, 18 + 0.5, 1],
        [22 + 0.5, 19 + 0.5, 1],
        [22 + 0.5, 20 + 0.5, 1],
        [22 + 0.5, 21 + 0.5, 1],
        [22 + 0.5, 22 + 0.5, 1],
        [22 + 0.5, 23 + 0.5, 1],
        [23 + 0.5, 23 + 0.5, 1],
        [24 + 0.5, 23 + 0.5, 1],
        [25 + 0.5, 23 + 0.5, 1],
    ]

    # 使用不同的位置列表创建障碍物
    obstacle_ids_1 = create_obstacle(physicsClient, obstacle_positions_set1)
    grid_size_x = 30
    grid_size_y = 30
    grid_map = create_grid_map2(physicsClient, grid_size_x, grid_size_y)

    print("生成的网格地图 (0: 空闲, 1: 障碍物):")
    for row in grid_map:
        print(row)

    # import matplotlib.pyplot as plt
    # # origin='lower' 确保左下角为原点
    # plt.imshow(grid_map, origin='lower', interpolation='nearest', cmap='gray')
    # plt.title('Grid Map')
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.colorbar(label='0: empty, 1: obstacle')
    # plt.show()

    # 定义 A* 算法的起始点和目标点 (网格坐标)
    start_pos = (2, 2)  # 第 5 行，第 5 列 (从 0 开始计数)
    goal_pos = (28, 28)  # 第 28 行，第 28 列

    # 使用 A* 算法生成路径
    path = a_star_pathfinding(grid_map, start_pos, goal_pos)

    smoothed_path_grid = None  # 初始化平滑路径网格坐标
    smoothed_path_world = None  # 初始化平滑路径世界坐标

    if path:

        # 进行路径平滑
        path_world_coords = []  # 将网格坐标路径转换为世界坐标路径，用于 Bezier 曲线平滑
        for grid_pos in path:
            world_x = grid_pos[1] * 1.0 + 1.0 / 2.0
            world_y = grid_pos[0] * 1.0 + 1.0 / 2.0
            path_world_coords.append([world_x, world_y])

        smoothed_path_world = smooth_path_bezier(path_world_coords)  # 平滑后的路径 (世界坐标)

        smoothed_path_grid = []  # 将平滑后的世界坐标路径转换回网格坐标，用于可视化 (可选，如果需要网格坐标的平滑路径)
        for world_pos in smoothed_path_world:
            grid_col = int(world_pos[0] / 1.0)
            grid_row = int(world_pos[1] / 1.0)
            smoothed_path_grid.append((grid_row, grid_col))

        print("找到路径：", path)
        print("平滑后的路径：", smoothed_path_grid)
        visualize_path(physicsClient, path)  # 可视化原始路径
        visualize_path(physicsClient, path, smooth_path=smoothed_path_world)  # 同时可视化原始路径和平滑路径 (平滑路径为蓝色)

    else:
        print("未能找到路径。")

    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    while True:
        p.stepSimulation()
        time.sleep(1./240.)
