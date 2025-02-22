import pybullet as p
import random
import math
import os
import platform
import numpy as np
from playground.pure_pursuit.PurePursuitController import PurePursuitController



class ZBicycleFinal:
    def __init__(self, client, pos=None, orn=None, obstacle_ids=[]):
        if orn is None:
            orn = [0, 0, 1.57]
        if pos is None:
            pos = [-4, -1, 1]
        self.client = client
        system = platform.system()
        if system == "Windows":
            f_name = os.path.join(os.path.dirname(__file__), 'bicycle_urdf\\bike.xml')
        else:
            f_name = os.path.join(os.path.dirname(__file__), 'bicycle_urdf/bike.xml')
        startOrientation = p.getQuaternionFromEuler(orn)
        self.bicycleId = p.loadURDF(fileName=f_name, basePosition=pos, baseOrientation=startOrientation)
        self.handlebar_joint = 0
        self.front_wheel_joint = 1
        self.back_wheel_joint = 2
        self.fly_wheel_joint = 4
        self.gyros_link = 5
        self.MAX_FORCE = 2000
        self.proximity_threshold = 0.3
        self.bicycle_vel = 0.8
        self.pure_pursuit_controller = None

        self.num_rays = 180
        self.ray_len = 12.0
        self.lidar_origin_offset = [0., 0.55, .7]  # 激光雷达相对于小车的位置偏移量
        self.initial_joint_positions = None
        self.initial_joint_velocities = None
        self.initial_position, self.initial_orientation = p.getBasePositionAndOrientation(self.bicycleId)
        self.update_frequency = 3  # 每 n 帧更新一次
        self.frame_count = 0
        # 初始化 last_lidar_info 用于在不更新激光雷达信息的帧中，仍然提供一个值，避免程序出错
        self.last_lidar_info = np.full(self.num_rays, self.ray_len, dtype=np.float32)

        self.obstacle_ids = obstacle_ids
        self.sin_array = [self.ray_len * math.sin(2. * math.pi * float(i) / self.num_rays) for i in
                          range(self.num_rays)]
        self.cos_array = [self.ray_len * math.cos(2. * math.pi * float(i) / self.num_rays) for i in
                          range(self.num_rays)]

        # 设置飞轮速度上限
        p.changeDynamics(self.bicycleId,
                         self.fly_wheel_joint,
                         maxJointVelocity=120.0,
                         physicsClientId=self.client)

    def init_pure_pursuit_controller(self, bicycle_object, lookahead_distance=2.5, wheelbase=1.2):
        self.pure_pursuit_controller = PurePursuitController(bicycle_object,
                                                             lookahead_distance=lookahead_distance,
                                                             wheelbase=wheelbase)
        
    def apply_action(self, action):
        """
        Apply the action to the bicycle.

        Parameters:
        action[0]控制车把位置
        action[1]控制前后轮速度
        action[2]控制飞轮
        """
        pure_pursuit_action, _ = self.pure_pursuit_controller.get_control_action(action)
        # action[0] = frame_to_handlebar 车把位置控制
        p.setJointMotorControl2(bodyUniqueId=self.bicycleId,
                                jointIndex=self.handlebar_joint,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=pure_pursuit_action[0],
                                force=self.MAX_FORCE,
                                physicsClientId=self.client)

        # action[1] = handlebar_to_frontwheel 前轮速度控制
        p.setJointMotorControl2(bodyUniqueId=self.bicycleId,
                                jointIndex=self.front_wheel_joint,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=self.bicycle_vel,
                                force=self.MAX_FORCE,
                                physicsClientId=self.client)

        # action[1] = frame_to_backwheel 后轮速度控制
        p.setJointMotorControl2(bodyUniqueId=self.bicycleId,
                                jointIndex=self.back_wheel_joint,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=self.bicycle_vel,
                                force=self.MAX_FORCE,
                                physicsClientId=self.client)

        # action[2] = flyWheelLink_to_flyWheel 飞轮控制
        p.setJointMotorControl2(bodyUniqueId=self.bicycleId,
                                jointIndex=self.fly_wheel_joint,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=pure_pursuit_action[2],
                                force=self.MAX_FORCE,
                                physicsClientId=self.client)

    def get_observation(self):
        self.frame_count += 1
        # Get the position位置 and orientation方向(姿态) of the bicycle in the simulation
        pos, _ = p.getBasePositionAndOrientation(self.bicycleId, self.client)
        # The rotation order is first roll around X, then pitch around Y and finally yaw around Z
        # 将四元数转换为欧拉角
        # euler_angles = p.getEulerFromQuaternion(orn)
        # 获取偏航角（yaw）
        # yaw = euler_angles[2]
        # roll_angle = ang[0]
        # p.getBaseVelocity()返回的格式 (线速度(x, y, z), 角速度(wx, wy, wz))
        # _, angular_velocity = p.getBaseVelocity(self.bicycleId, self.client)
        # roll_vel = angular_velocity[0]

        gyros_link_state = p.getLinkState(self.bicycleId, self.gyros_link, computeLinkVelocity=1)
        # gyros_link_position = gyros_link_state[0]
        gyros_link_orientation = gyros_link_state[1]
        link_ang = p.getEulerFromQuaternion(gyros_link_orientation)
        roll_angle = link_ang[0]
        yaw_angle = link_ang[2]
        gyros_link_angular_vel = gyros_link_state[7]
        roll_angular_vel = gyros_link_angular_vel[0]

        handlebar_joint_state = p.getJointState(self.bicycleId, self.handlebar_joint, self.client)
        handlebar_joint_ang = handlebar_joint_state[0]
        handlebar_joint_vel = handlebar_joint_state[1]

        back_wheel_joint_state = p.getJointState(self.bicycleId, self.back_wheel_joint, self.client)
        # back_wheel_joint_ang = back_wheel_joint_state[0]
        back_wheel_joint_vel = back_wheel_joint_state[1]

        fly_wheel_joint_state = p.getJointState(self.bicycleId, self.fly_wheel_joint, self.client)
        # fly_wheel_joint_ang = fly_wheel_joint_state[0] % (2 * math.pi)
        fly_wheel_joint_vel = fly_wheel_joint_state[1]

        lidar_info = None  # 初始化 lidar_info
        is_collided, is_proximity = False, False
        if self.frame_count % self.update_frequency == 0:
            lidar_info = self._get_lidar_info4(pos, yaw_angle)
            is_collided, is_proximity = self._is_collided_and_proximity()

        observation = [pos[0], pos[1], yaw_angle,
                       roll_angle, roll_angular_vel,
                       handlebar_joint_ang, handlebar_joint_vel,
                       back_wheel_joint_vel, fly_wheel_joint_vel,
                       lidar_info if lidar_info is not None else self.last_lidar_info,  # 使用上次的激光雷达信息
                       is_collided,
                       is_proximity
                       ]

        # 保存激光雷达信息，下次使用
        self.last_lidar_info = lidar_info if lidar_info is not None else self.last_lidar_info

        return observation

    def reset(self):
        p.resetBasePositionAndOrientation(self.bicycleId, self.initial_position, self.initial_orientation)
        p.resetJointState(self.bicycleId, self.handlebar_joint, targetValue=0, targetVelocity=0)
        p.resetJointState(self.bicycleId, self.fly_wheel_joint, targetValue=0, targetVelocity=0)
        p.resetJointState(self.bicycleId, self.front_wheel_joint, targetValue=0, targetVelocity=0)
        p.resetJointState(self.bicycleId, self.back_wheel_joint, targetValue=0, targetVelocity=0)
        self.last_lidar_info = np.full(self.num_rays, self.ray_len, dtype=np.float32)  # 初始化 last_lidar_info
        self.frame_count = 0
        return self.get_observation()

    def _is_collided_and_proximity(self):
        """
        检测机器人是否与任何障碍物发生真实碰撞 (使用 getContactPoints)
        检测机器人是否接近障碍物 (使用 getClosestPoints)，返回true为太靠近，false为不是很靠近
        返回碰撞检测和接近检测结果
        """
        for obstacle_id in self.obstacle_ids:
            contact_points = p.getContactPoints(bodyA=self.bicycleId, bodyB=obstacle_id, physicsClientId=self.client)
            closest_points = p.getClosestPoints(bodyA=self.bicycleId, bodyB=obstacle_id,
                                                distance=self.proximity_threshold, physicsClientId=self.client)
            is_collided = len(contact_points) > 0  # getContactPoints 返回的列表不为空，表示有接触点，即发生碰撞
            is_proximity = len(closest_points) > 0  # getClosestPoints 返回的列表不为空，表示在阈值距离内有最近点

            if is_collided:
                return True, True
            elif is_collided is False and is_proximity is True:
                return False, True
        return False, False

    def _get_lidar_info4(self, bicycle_pos, bicycle_yaw):
        # 计算射线起点的坐标
        ray_start = np.array(bicycle_pos) + np.array(self.lidar_origin_offset)

        # 创建射线起点和终点的数组
        rayFrom = np.tile(ray_start, (self.num_rays, 1))
        rayTo = np.zeros((self.num_rays, 3))  # 初始化 rayTo

        # 计算每条射线的角度，并根据yaw角进行调整
        start_angle = bicycle_yaw - math.pi / 2  # 扫描范围的起始角度
        for i in range(self.num_rays):
            angle = start_angle + (math.pi * float(i) / (self.num_rays - 1))  # 180度范围内的角度
            rayTo[i] = rayFrom[i] + [self.ray_len * math.cos(angle), self.ray_len * math.sin(angle), 0]

        results = p.rayTestBatch(rayFromPositions=rayFrom.tolist(), rayToPositions=rayTo.tolist())

        # p.removeAllUserDebugItems()
        # for i, result in enumerate(results):
            # if i == 0:
            #     p.addUserDebugLine(rayFrom[i], rayTo[i], lineColorRGB=[1, 0, 0], lineWidth=2.0)
            # elif i == 30:
            #     p.addUserDebugLine(rayFrom[i], rayTo[i], lineColorRGB=[0, 1, 0], lineWidth=2.0)
            # elif i == 60:
            #     p.addUserDebugLine(rayFrom[i], rayTo[i], lineColorRGB=[0, 0, 1], lineWidth=2.0)
            # elif i == 90:
            #     p.addUserDebugLine(rayFrom[i], rayTo[i], lineColorRGB=[1, 1, 1], lineWidth=2.0)
            # elif i == 120:
            #     p.addUserDebugLine(rayFrom[i], rayTo[i], lineColorRGB=[0, 0, 0], lineWidth=2.0)
            # if result[0] < 0:
            #     p.addUserDebugLine(rayFrom[i], rayTo[i], lineColorRGB=[0, 1, 0], lineWidth=1.0)
            # else:
            #     hit_position = result[3]
            #     # 计算击中点到射线起点的距离
            #     distance = ((hit_position[0] - rayFrom[i][0]) ** 2 +
            #                 (hit_position[1] - rayFrom[i][1]) ** 2 +
            #                 (hit_position[2] - rayFrom[i][2]) ** 2) ** 0.5
            #     # 在击中点附近显示距离
            #     text_position = (hit_position[0], hit_position[1], hit_position[2] + 0.1)  # 提高文本显示位置
            #     p.addUserDebugText(f"{distance:.2f} m", text_position, textColorRGB=[1, 1, 1], textSize=1.0)
            #     # 显示击中点的射线
            #     p.addUserDebugLine(rayFrom[i], hit_position, lineColorRGB=[1, 0, 0], lineWidth=1.0)

        # 计算距离
        distance = np.array([
            self.ray_len if res[0] < 0 else np.linalg.norm(np.array(res[3]) - ray_start)
            for res in results
        ], dtype=np.float32)

        # distance_reshaped = distance.reshape(60, 3)  # 使用reshape将其变为(60, 3)的形状，方便每3个元素进行平均
        # averaged_distance = np.mean(distance_reshaped, axis=1, keepdims=True).flatten().tolist()  # 对每一行取平均值
        # p.removeAllUserDebugItems()
        # for i in range(0, 60, 10):
        #     result = averaged_distance[i]
        #     angle = start_angle + (math.pi * float(i) / (60 - 1))  # 180度范围内的角度
        #     rayTo[i] = rayFrom[i] + [result * math.cos(angle), result * math.sin(angle), 0]
        #     p.addUserDebugLine(rayFrom[i], rayTo[i], lineColorRGB=[1, 0, 0], lineWidth=1.0)
        return distance
