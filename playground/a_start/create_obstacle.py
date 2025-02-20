import pybullet as p
import math


def create_obstacle(client_id):
    """
    创建一个或多个障碍物。

    参数:
        client_id: PyBullet 客户端 ID
        obstacle_positions: 障碍物位置列表，每个位置是一个列表 [x, y, z]

    返回:
        障碍物的 bodyId 列表
    """

    # 定义不同的障碍物位置列表
    obstacle_positions1 = [
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

    obstacle_positions2 = [
        [3 + 0.5, 8 + 0.5, 1],
        [3 + 0.5, 9 + 0.5, 1],
        [4 + 0.5, 8 + 0.5, 1],
        [4 + 0.5, 9 + 0.5, 1],

        [-2 + 0.5, 12 + 0.5, 1],
        [-2 + 0.5, 13 + 0.5, 1],
        [-1 + 0.5, 12 + 0.5, 1],
        [-1 + 0.5, 13 + 0.5, 1],

        [11 + 0.5, 6 + 0.5, 1],
        [11 + 0.5, 7 + 0.5, 1],
        [12 + 0.5, 6 + 0.5, 1],
        [12 + 0.5, 7 + 0.5, 1],

        [15 + 0.5, 12 + 0.5, 1],
        [15 + 0.5, 13 + 0.5, 1],
        [16 + 0.5, 12 + 0.5, 1],
        [16 + 0.5, 13 + 0.5, 1],

        [9 + 0.5, 15 + 0.5, 1],
        [9 + 0.5, 16 + 0.5, 1],
        [10 + 0.5, 15 + 0.5, 1],
        [10 + 0.5, 16 + 0.5, 1],
    ]

    collision_shape = p.createCollisionShape(p.GEOM_CYLINDER, halfExtents=[0.5, 0.5, 1.0], radius=1.5, height=3.0)
    visual_shape = p.createVisualShape(p.GEOM_CYLINDER, halfExtents=[0.5, 0.5, 1.0],
                                       rgbaColor=[0.92, 0.94, 0.94, 1], radius=1.5, length=3.0)
    obstacle_ids = []
    obstacle_positions1 = generate_obstacle_positions_from_file()
    for obstacle_pos in obstacle_positions1:
        obstacle_id = p.createMultiBody(baseCollisionShapeIndex=collision_shape,
                                        baseVisualShapeIndex=visual_shape,
                                        basePosition=obstacle_pos,
                                        physicsClientId=client_id)
        obstacle_ids.append(obstacle_id)
    # 根据你的需求创建四面墙
    left = -14
    right = 14
    top = 26
    bottom = -2
    left_wall = create_wall(left, bottom, left, top)
    right_wall = create_wall(right, bottom, right, top)
    top_wall = create_wall(left, top, right, top)
    bottom_wall = create_wall(left, bottom, right, bottom)
    obstacle_ids.append(left_wall)
    obstacle_ids.append(right_wall)
    obstacle_ids.append(left_wall)
    obstacle_ids.append(top_wall)
    obstacle_ids.append(bottom_wall)

    return obstacle_ids

# 生成四面墙
def create_wall(x1, y1, x2, y2):
    # 墙的参数
    wall_thickness = 1  # 墙的厚度
    wall_height = 3  # 墙的高度，可根据需要调整
    distance = ((x2 - x1) ** 2 + ((y2 - y1) ** 2)) ** 0.5
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    angle = math.atan2(y2 - y1, x2 - x1)

    # 创建墙的形状
    wall_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[distance / 2, wall_thickness / 2, wall_height / 2])
    wall_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[distance / 2, wall_thickness / 2, wall_height / 2], rgbaColor=[0.5, 0.5, 0.5, 1]) # 可设置颜色

    # 创建墙的刚体
    wall_id = p.createMultiBody(0, wall_collision, wall_visual, [center_x, center_y, wall_height / 2], p.getQuaternionFromEuler([0, 0, angle]))
    return wall_id


def generate_obstacle_positions_from_file(
        file_path='/home/chen/denghang/bicycle-rl/playground/a_start/obstacle_map.txt'):
    """
    从图画文件读取障碍物位置并生成 obstacle_positions2 列表。
    **修改：**
    1.  **坐标原点：** txt 最后一行中间为 (0, 0)。
    2.  **X 轴：** 左负右正。
    3.  **Y 轴：** 向上为正。
    4.  **单位距离：** 每个 'O' 或 'X' 代表一个单位。
    5.  **文本宽度：** 假定每行文本宽度为 30。

    Args:
        file_path (str): 图画文件的路径。

    Returns:
        list: obstacle_positions2 列表。
    """
    obstacle_positions = []
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
            lines.reverse()  # 反转行列表，使文件最后一行成为列表第一行

            text_width = 30  # 每行文本宽度固定为 30
            center_col_index = text_width // 2  # 计算中心列索引 (整数除法)

            for row_index, line in enumerate(lines):
                for col_index, char in enumerate(line.strip()):  # strip() 移除行尾换行符
                    if char == 'X':
                        # 计算相对于原点的 x 和 y 坐标
                        x_coord = col_index - center_col_index  # X 轴：左负右正
                        y_coord = row_index  # Y 轴：向上为正，行索引直接作为 y 坐标

                        obstacle_positions.append(
                            [x_coord + 0.5, y_coord + 0.5, 1.0])  # [x, y, z]  z=1 表示障碍物高度，+0.5 偏移到格子中心
    except FileNotFoundError:
        print(f"错误：文件 {file_path} 未找到！请确保文件路径正确。")
        return None
    return obstacle_positions
