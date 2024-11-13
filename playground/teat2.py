import numpy as np
import matplotlib.pyplot as plt


def create_radar_image(ray_results, grid_size=64, max_distance=10):
    """
    根据 rayTestBatch 的返回值创建一个二维雷达图像。

    ray_results: rayTestBatch 的返回结果
    grid_size: 图像的大小，表示雷达图的分辨率
    max_distance: 最大可检测的距离（用于归一化）

    返回: 一个二维图像（雷达图）
    """
    # 创建一个空的雷达图像，初始化为全0
    radar_image = np.zeros((grid_size, grid_size))

    # 假设射线分布在0到360度之间，均匀分布
    num_rays = len(ray_results)
    angle_step = 360 / num_rays  # 每条射线对应的角度间隔

    for i, result in enumerate(ray_results):
        # 获取命中位置 (hitPosition)
        hit_position = np.array(result[3])  # hitPosition 是一个包含x, y, z的列表

        # 计算该射线的距离，假设自行车的当前位置是原点 (0, 0)
        distance = np.linalg.norm(hit_position[:2])  # 只考虑x和y坐标

        # 将距离归一化到 [0, max_distance]
        normalized_distance = min(distance, max_distance) / max_distance

        # 计算射线对应的角度
        angle = i * angle_step  # 当前射线的角度

        # 将角度转换为图像中的坐标系
        # 假设自行车处于图像的中心 (grid_size // 2, grid_size // 2)
        center_x, center_y = grid_size // 2, grid_size // 2
        x = int(center_x + normalized_distance * np.cos(np.deg2rad(angle)) * (grid_size // 2))
        y = int(center_y + normalized_distance * np.sin(np.deg2rad(angle)) * (grid_size // 2))

        # 将该点的值设置为1，表示该位置有障碍物
        if 0 <= x < grid_size and 0 <= y < grid_size:
            radar_image[x, y] = 1

    return radar_image


# 示例：生成一个包含1000条射线的雷达图像
ray_results = [
    # 模拟的 rayTestBatch 返回值示例
    [None, None, None, [np.cos(np.deg2rad(i)) * 5, np.sin(np.deg2rad(i)) * 5, 0], None]
    for i in range(360)  # 假设有360条射线
]

# 创建雷达图像
radar_image = create_radar_image(ray_results, grid_size=64, max_distance=10)

# 显示图像
plt.imshow(radar_image, cmap='gray')
plt.colorbar()
plt.title("Radar Image")
plt.show()
