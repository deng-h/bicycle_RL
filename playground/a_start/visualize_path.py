import numpy as np
import pybullet as p

def bezier_curve(control_points, n_segments=10):
    """
    生成 Bezier 曲线上的点。

    参数:
        control_points: 控制点坐标列表 (NumPy 数组或列表)
        n_segments:  Bezier 曲线分段数 (控制曲线平滑度和点数量)

    返回:
        Bezier 曲线上的点坐标列表 (NumPy 数组)
    """
    if not control_points:
        return []

    control_points = np.array(control_points)
    curve_points = []
    for i in range(n_segments + 1):
        t = i / n_segments
        point = np.zeros(2) # 假设是 2D 坐标
        n = len(control_points) - 1
        for j, cp in enumerate(control_points):
            point += binomial_coefficient(n, j) * (1 - t)**(n - j) * t**j * cp
        curve_points.append(point)
    return np.array(curve_points)


def binomial_coefficient(n, k):
    """
    计算二项式系数 (n choose k)。
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    if k > n // 2:
        k = n - k

    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def smooth_path_bezier(path, segment_length=5):
    """
    使用 Bezier 曲线平滑路径。

    参数:
        path:  原始路径点坐标列表 (NumPy 数组或列表)
        segment_length:  每段 Bezier 曲线使用的路径点数量 (控制平滑程度)

    返回:
        平滑后的路径点坐标列表 (NumPy 数组)
    """
    if len(path) <= 2: # 路径点太少，无法平滑
        return path

    smoothed_path = []
    for i in range(0, len(path) - 1, segment_length - 1): # 步长为 segment_length - 1，保证每段曲线有 segment_length 个控制点
        segment_points = path[i:min(i + segment_length, len(path))] # 取一段路径点作为控制点
        curve = bezier_curve(segment_points) # 生成 Bezier 曲线
        smoothed_path.extend(curve)

    smoothed_path.append(path[-1]) # 添加最后一个点，确保路径完整性
    return np.array(smoothed_path)


def visualize_path(client_id, path, resolution=1.0, smooth_path=None):
    """
    在 PyBullet 中可视化路径。

    参数:
        client_id: PyBullet 客户端 ID
        path: 路径节点坐标列表 (网格坐标)
        resolution: 每个网格单元代表的物理尺寸
    """
    if path is None:
        print("未找到路径，无法可视化。")
        return

    line_color = [0, 1, 0]  # 绿色
    line_width = 3
    points = []
    for grid_pos in path:
        # 将网格坐标转换为世界坐标 (网格中心点)
        world_x = grid_pos[1] * resolution + resolution / 2.0
        world_y = grid_pos[0] * resolution + resolution / 2.0
        world_z = 0.5  # 路径线的高度
        points.append([world_x, world_y, world_z])

    if smooth_path is not None:  # 如果提供了平滑路径，则可视化平滑路径
        smooth_line_color = [0, 0, 1]  # 蓝色，平滑路径颜色
        smooth_points = []
        for grid_pos in smooth_path: # smooth_path 已经是世界坐标
            smooth_points.append([grid_pos[0], grid_pos[1], 0.55]) # 平滑路径稍稍抬高，避免与原始路径重叠

        for i in range(len(smooth_points) - 1):
            p.addUserDebugLine(smooth_points[i], smooth_points[i+1], smooth_line_color, lineWidth=line_width,
                               physicsClientId=client_id)
    else:
        # 使用 PyBullet 的 addUserDebugLine 绘制路径线段
        for i in range(len(points) - 1):
            p.addUserDebugLine(points[i], points[i + 1], line_color, lineWidth=line_width, physicsClientId=client_id)