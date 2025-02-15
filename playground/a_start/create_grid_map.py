import numpy as np
import pybullet as p

def create_grid_map(client_id, grid_size_x, grid_size_y, resolution=1.0):
    """
    创建网格地图，标记被障碍物占据的网格单元。

    参数:
        client_id: PyBullet 客户端 ID
        grid_size_x: 网格在 X 轴方向的大小
        grid_size_y: 网格在 Y 轴方向的大小
        resolution: 每个网格单元代表的物理尺寸 (默认 1.0)

    返回:
        一个二维 NumPy 数组，表示网格地图。0 代表空闲区域, 1 代表障碍物。
    """

    grid_map = np.zeros((grid_size_y, grid_size_x), dtype=int)
    num_rows, num_cols = grid_map.shape

    for row in range(num_rows):
        for col in range(num_cols):
            world_x = col * resolution + resolution / 2.0  # 网格单元中心点的 X 坐标
            world_y = row * resolution + resolution / 2.0  # 网格单元中心点的 Y 坐标

            # 检查该中心点是否与任何障碍物发生碰撞
            for body_id in range(p.getNumBodies(physicsClientId=client_id)):
                if body_id == 0:  # 跳过地面 (假设地面 bodyId 为 0)
                    continue # 如果你的环境中地面不是 bodyId 0, 需要修改判断条件

                aabb_min, aabb_max = p.getAABB(body_id, physicsClientId=client_id) # 获取物体的 AABB

                # 检查中心点是否在障碍物的 AABB 范围内 (近似碰撞检测)
                if (aabb_min[0] <= world_x <= aabb_max[0] and
                    aabb_min[1] <= world_y <= aabb_max[1]):
                    grid_map[row, col] = 1  # 标记为障碍物
                    break  # 如果已检测到碰撞，则跳出内层循环，继续下一个网格单元

    return grid_map


def create_grid_map2(client_id, grid_size_x, grid_size_y, resolution=1.0, inflation_radius=1): # 添加 inflation_radius 参数
    """
    创建网格地图，标记被障碍物占据的网格单元, 并进行障碍物膨胀。

    参数:
        ... (其他参数不变) ...
        inflation_radius: 障碍物膨胀半径 (单位：网格单元格)
    """
    grid_map = np.zeros((grid_size_y, grid_size_x), dtype=int)
    num_rows, num_cols = grid_map.shape

    for row in range(num_rows):
        for col in range(num_cols):
            world_x = col * resolution + resolution / 2.0
            world_y = row * resolution + resolution / 2.0

            for body_id in range(p.getNumBodies(physicsClientId=client_id)):
                if body_id == 0:
                    continue

                aabb_min, aabb_max = p.getAABB(body_id, physicsClientId=client_id)

                if (aabb_min[0] <= world_x <= aabb_max[0] and
                        aabb_min[1] <= world_y <= aabb_max[1]):
                    grid_map[row, col] = 1  # 标记为障碍物

                    # 障碍物膨胀处理
                    for i in range(max(0, row - inflation_radius), min(num_rows, row + inflation_radius + 1)):
                        for j in range(max(0, col - inflation_radius), min(num_cols, col + inflation_radius + 1)):
                            if grid_map[i, j] == 0: # 避免重复标记，只膨胀空闲区域
                                grid_map[i, j] = 1
                    break

    return grid_map
