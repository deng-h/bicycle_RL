import csv

import pybullet as p
import numpy as np
import math
from bicycle_dengh.resources.bicycle import Bicycle
import pybullet_data
import time
from simple_pid import PID
from playground.a_start.create_grid_map import create_grid_map2
from playground.a_start.create_obstacle import create_obstacle
from playground.a_start.a_star_algo import a_star_pathfinding
from playground.a_start.visualize_path import smooth_path_bezier, visualize_path
from playground.a_start.get_goal_pos import get_goal_pos
from bicycle_dengh.resources.goal import Goal



class PurePursuitController:
    def __init__(self, bicycle, lookahead_distance=2.5, wheelbase=1.4):
        self.bicycle = bicycle
        self.lookahead_distance = lookahead_distance
        self.wheelbase = wheelbase  # 自行车轴距，需要根据你的自行车模型调整
        self.roll_angle_pid = PID(1200, 750, 40, setpoint=0.0)

    def set_lookahead_distance(self, lookahead_distance):
        """动态调整lookahead distance"""
        self.lookahead_distance = lookahead_distance

    def calculate_steering_angle(self, current_position, current_yaw, path_points):
        """
        计算Pure Pursuit算法的转向角.

        参数:
        current_position: 当前自行车的位置 [x, y]
        current_yaw: 当前自行车的偏航角 (弧度)
        path_points: 规划路径点列表，每个点格式为 [x, y]

        返回值:
        steering_angle:  车把的期望转角 (弧度)
        lookahead_point:  Lookahead point 的坐标 [x, y]
        """
        lookahead_point = None

        # 1. 寻找Lookahead Point
        min_dist_index = 0
        min_dist = float('inf')
        for i in range(len(path_points)):
            dist = np.linalg.norm(np.array(current_position) - np.array(path_points[i][:2]))
            if dist < min_dist:
                min_dist = dist
                min_dist_index = i

        for i in range(min_dist_index, len(path_points)):  # 从最近点索引开始向后查找
            point = path_points[i]
            dist_to_point = np.linalg.norm(np.array(current_position) - np.array(point[:2]))
            if dist_to_point >= self.lookahead_distance:
                lookahead_point = point[:2]  # 找到lookahead point
                break

        if lookahead_point is None:
            # 如果没有找到足够远的点，就用最后一个点作为lookahead point
            lookahead_point = path_points[-1][:2]

        # 2. 将Lookahead Point转换到车辆坐标系下
        dx = lookahead_point[0] - current_position[0]
        dy = lookahead_point[1] - current_position[1]
        lookahead_in_vehicle_x = dx * math.cos(current_yaw) + dy * math.sin(current_yaw)
        lookahead_in_vehicle_y = -dx * math.sin(current_yaw) + dy * math.cos(current_yaw)

        # 3. 计算曲率和转向角 (使用自行车模型简化公式)
        alpha = math.atan2(lookahead_in_vehicle_y, lookahead_in_vehicle_x) # 期望方向角 (车辆坐标系下)

        # 简化 Pure Pursuit 公式 (基于运动学自行车模型近似)
        steering_angle = math.atan2((2 * self.wheelbase * math.sin(alpha)), max(1e-6, self.lookahead_distance) ) # 避免除以零

        return steering_angle, lookahead_point


    def get_control_action(self, path_points, current_velocity=2.0): # 可以设置期望速度
        """
        根据Pure Pursuit算法计算控制动作.

        参数:
        path_points: 规划路径点列表
        current_velocity:  期望的自行车速度 (可以根据需要调整)

        返回值:
        action:  控制动作 [handlebar_angle, wheel_velocity, flywheel_velocity]
        lookahead_point:  Lookahead point 坐标 [x, y]
        """
        # observation = self.bicycle.get_observation()
        observation = self.bicycle.get_observation()
        current_position = [observation[0], observation[1]]  # x, y 位置
        current_yaw = observation[2]  # 偏航角 yaw_angle
        roll_angle = observation[3]  # 翻滚角 roll_angle
        roll_angle_control = self.roll_angle_pid(roll_angle)

        steering_angle, lookahead_point = self.calculate_steering_angle(current_position, current_yaw, path_points)

        #  控制量映射到自行车Action
        target_handlebar_angle = steering_angle  # 直接将计算出的转向角作为车把目标角度
        target_wheel_velocity = current_velocity  # 设定恒定速度 (或者可以根据误差动态调整速度)
        target_flywheel_velocity = roll_angle_control

        action = [target_handlebar_angle, target_wheel_velocity, -target_flywheel_velocity]
        return action, lookahead_point

    def get_roll_angle_control(self):
        observation = self.bicycle.get_observation_simple()
        roll_angle = observation[3]
        roll_angle_control = self.roll_angle_pid(roll_angle)
        return -roll_angle_control

def get_path(client, grid_map, bicycle_start_pos, goal_pos):
    path = a_star_pathfinding(grid_map, bicycle_start_pos, goal_pos)
    # path格式为[(row, col), (row, col), ...]
    smoothed_path_world = None  # 初始化平滑路径世界坐标
    if path:
        path_world_coords = []  # 将网格坐标路径转换为世界坐标路径，用于 Bezier 曲线平滑
        for grid_pos in path:
            world_x = grid_pos[1] * 1.0 + 1.0 / 2.0
            world_y = grid_pos[0] * 1.0 + 1.0 / 2.0
            path_world_coords.append([world_x, world_y])

        # 平滑后的路径 (世界坐标)
        smoothed_path_world = smooth_path_bezier(path_world_coords, segment_length=25)
        visualize_path(client, path, smooth_path=smoothed_path_world)
        # --- 添加基于距离的抽稀代码 ---
        # min_dist = 3.0  # 最小距离阈值
        # after_sampled_path = []  # 初始化采样后的路径
        # last_point = None
        # for point in smoothed_path_world:
        #     if last_point is None:
        #         after_sampled_path.append(point)  # 第一个点总是保留
        #         last_point = point
        #     else:
        #         dist = np.linalg.norm(np.array(point) - np.array(last_point))
        #         if dist > min_dist:
        #             after_sampled_path.append(point)
        #             last_point = point
        # smoothed_path_world = after_sampled_path
        # --- 抽稀代码结束 ---
    return smoothed_path_world


def draw_trajectory_fun(client, obs, trajectory_points, trajectory_lines):
    current_pos = (obs[0], obs[1])
    trajectory_points.append(current_pos)

    if len(trajectory_points) > 1:
        prev_pos = trajectory_points[-2]
        line_id = p.addUserDebugLine(
            lineFromXYZ=[prev_pos[0], prev_pos[1], 0.01],  # 起始点，z坐标稍微抬高
            lineToXYZ=[current_pos[0], current_pos[1], 0.01],  # 结束点，z坐标稍微抬高
            lineColorRGB=[1, 0, 0],  # 红色
            lineWidth=2,
            physicsClientId=client
        )
        trajectory_lines.append(line_id)


if __name__ == '__main__':
    client = p.connect(p.GUI)
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
    p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
    # 设置俯视视角
    # p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[15, 12, 10])
    p.setGravity(0, 0, -10)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    plane_id = p.loadURDF("plane.urdf")
    p.changeDynamics(plane_id, -1, lateralFriction=3.5)  # -1表示所有部件
    p.setRealTimeSimulation(0)  # 关闭实时仿真，手动step
    time_step = 1. / 24.
    p.setTimeStep(time_step)
    bicycle = Bicycle(client, 120.0)
    pure_pursuit_controller = PurePursuitController(bicycle, lookahead_distance=2.0, wheelbase=1.2) # 创建控制器实例
    # create_obstacle(client)
    # grid_map = create_grid_map2(client, 30, 30)
    # goal_pos = get_goal_pos()
    # goal_id = Goal(client, goal_pos)
    # smoothed_path_world = get_path(client=client, grid_map=grid_map, bicycle_start_pos=(1, 1), goal_pos=goal_pos)
    # zeros_column = np.zeros((smoothed_path_world.shape[0], 1))  # 创建一个形状为 (行数, 1) 的全零数组
    # planned_path = np.hstack((smoothed_path_world, zeros_column))  # 将全零数组与原始数组水平拼接
    # print(smoothed_path_world)
    # print(planned_path)
    draw_trajectory = True
    trajectory_points = []
    trajectory_lines = []
    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
    # -----  示例路径点 (你需要替换为你A*算法生成的路径点) -----
    # planned_path = [
    #     [-1, 1.5, 0],
    #     [0, 1.5, 0],
    #     [1, 1.5, 0],
    # ]

    path_index = 0  # 路径点索引
    target_threshold = 3.0  # 到达目标点的阈值

    roll_angle_array = []  # 存储翻滚角数据
    for step in range(1000):
        current_pos = bicycle.get_observation()[:2]
        action, lookahead_point = pure_pursuit_controller.get_control_action([(1, 1), (2,2)])  # Pure Pursuit 控制
        # bicycle.apply_action(action)
        bicycle.apply_action([0, 0, action[2]])  # 仅控制飞轮
        p.stepSimulation()
        roll_angle = bicycle.get_observation()[3]
        roll_angle_array.append(roll_angle)

        if 700 <= step <= 701:
            bicycle.apply_lateral_disturbance(35.0, [-0.3, -0.0, 1.5])

        #  -----  绘制轨迹  -----
        # if draw_trajectory:
        #     draw_trajectory_fun(client, bicycle.get_observation(), trajectory_points, trajectory_lines)

        #  -----  (可选) 可视化  -----
        #  Debug Lookahead Point (红色球)
        # if lookahead_point is not None:
        #     lookahead_visual_id = p.createVisualShape(shapeType=p.GEOM_SPHERE, radius=0.1, rgbaColor=[1, 0, 0, 1])
        #     lookahead_body_id = p.createMultiBody(baseMass=0,
        #                                           baseVisualShapeIndex=lookahead_visual_id,
        #                                           basePosition=[lookahead_point[0], lookahead_point[1], 0.1])
        #     if step % 100 == 0:  # 每10步更新一下，避免创建过多物体
        #         p.removeBody(lookahead_body_id)

        #  -----  检查是否到达目标点 (最后一个路径点)  -----
        # target_point = planned_path[-1][:2]  # 最后一个路径点作为目标点
        # distance_to_target = np.linalg.norm(np.array(current_pos) - np.array(target_point))
        # if distance_to_target < target_threshold and path_index >= len(planned_path) - 1 :
        #     print("到达目标点!")
        #     break
        #
        # keys = p.getKeyboardEvents()
        # if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
        #     time.sleep(10000)
        # elif ord('z') in keys and keys[ord('z')] & p.KEY_WAS_TRIGGERED:
        #     p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-89, cameraTargetPosition=[15, 12, 10])

        time.sleep(time_step)
        if step % 100 == 0:
            print(f"Step: {step}, Roll Angle: {roll_angle}")

    with open(
            'D:\data\\1-L\\9-bicycle\\bicycle-rl\exp_data\平衡实验数据处理\倾斜角实验\\roll_angle_pid_BN.csv',
            'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['roll_angle'])  # 写入 CSV 文件的头部 写入列名，第一行
        # 将每个浮点数转换为包含该浮点数的列表
        rows = [[angle] for angle in roll_angle_array]
        csv_writer.writerows(rows)  # writerows 可以一次写入多行
        print("数据已保存到 roll_angle_pid.csv")

    p.disconnect()
