import pybullet as p
import pybullet_data
import time
import os
import numpy as np
from bicycle_dengh.resources.bicycle_lidar import BicycleLidar


if __name__ == '__main__':
    client = p.connect(p.GUI)  # 或 p.DIRECT
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -10)
    planeId = p.loadURDF("plane.urdf")

    # 创建 BicycleLidar 实例
    bicycle_lidar = BicycleLidar(client, max_flywheel_vel=10) # 假设 max_flywheel_vel 为 10

    # 设定安全范围半径 r
    safety_radius = 2.0  # 例如，设定安全半径为 2 米
    lidar_origin_offset = (0., 0., .7)  # 激光雷达相对于自行车的位置偏移
    while True:
        p.stepSimulation()
        time.sleep(1./240.)

        # 获取自行车位置
        bicycle_pos, _ = p.getBasePositionAndOrientation(bodyUniqueId=bicycle_lidar.bicycleId)

        # 清除之前的debug lines (可选，如果需要动态更新圆的位置)
        # p.removeAllUserDebugItems()

        # 绘制安全范围圆
        bicycle_lidar.draw_circle(np.array(bicycle_pos) + np.array(lidar_origin_offset), safety_radius, color=[0, 1, 0]) # 使用绿色表示安全范围

        # 获取激光雷达信息 (您可以继续调用您的雷达信息获取函数)
        distance = bicycle_lidar._get_lidar_info3(bicycle_pos)
        # ... (使用 distance 进行后续操作) ...

    p.disconnect()