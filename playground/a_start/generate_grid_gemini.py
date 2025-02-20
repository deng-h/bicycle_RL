import pybullet as p
import pybullet_data
import numpy as np
import time
from playground.a_start.a_star_algo import a_star_pathfinding
from playground.a_start.create_grid_map import create_grid_map2
from playground.a_start.visualize_path import visualize_path, smooth_path_bezier
from playground.a_start.create_obstacle import create_obstacle


if __name__ == '__main__':
    physicsClient = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    planeId = p.loadURDF("plane.urdf")

    # 设置俯视视角
    camera_distance = 10  # 相机距离
    camera_yaw = 0  # 相机偏航角（水平旋转）
    camera_pitch = -85  # 相机俯仰角（垂直旋转）
    camera_target_position = [15, 15, 10]  # 相机目标位置
    # 重置调试可视化相机
    p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)

    # 使用不同的位置列表创建障碍物
    obstacle_ids_1 = create_obstacle(physicsClient)
    grid_size_x = 30
    grid_size_y = 30
    grid_map = create_grid_map2(physicsClient, grid_size_x, grid_size_y)

    print("生成的网格地图 (0: 空闲, 1: 障碍物):")
    for row in grid_map:
        print(row)

    import matplotlib.pyplot as plt
    # origin='lower' 确保左下角为原点
    plt.imshow(grid_map, origin='lower', interpolation='nearest', cmap='gray')
    plt.title('Grid Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(label='0: empty, 1: obstacle')
    plt.show()

    # 定义 A* 算法的起始点和目标点 (网格坐标)
    start_pos = (2, 2)  # 第 5 行，第 5 列 (从 0 开始计数)
    goal_pos = (10, 12)  #  (20, 11)

    # 使用 A* 算法生成路径
    path = a_star_pathfinding(grid_map, start_pos, goal_pos)

    smoothed_path_grid = None  # 初始化平滑路径网格坐标
    smoothed_path_world = None  # 初始化平滑路径世界坐标

    if path:
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
