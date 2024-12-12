import pybullet as p
import math
import os
import numpy as np
import platform


def _map_to_image(ray_results, img_size=128, x_range=(-25, 25), y_range=(-3, 47)):
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


class BicycleLidar:
    def __init__(self, client, max_flywheel_vel, obstacle_ids=None):
        self.client = client

        system = platform.system()
        if system == "Windows":
            f_name = os.path.join(os.path.dirname(__file__), 'bicycle_urdf\\bike.xml')
        else:
            f_name = os.path.join(os.path.dirname(__file__), 'bicycle_urdf/bike.xml')

        startOrientation = p.getQuaternionFromEuler([0, 0, 1.57])
        self.bicycleId = p.loadURDF(fileName=f_name, basePosition=[0, 0, 1], baseOrientation=startOrientation,
                                    physicsClientId=self.client)
        # Number of joints: 7
        # jointIndex: 0 jointName: 'frame_to_handlebar'
        # jointIndex: 1 jointName: 'camera_joint'
        # jointIndex: 2 jointName: 'handlebar_to_frontwheel'
        # jointIndex: 3 jointName: 'frame_to_backwheel'
        # jointIndex: 4 jointName: 'frame_to_flyWheelLink'
        # jointIndex: 5 jointName: 'flyWheelLink_to_flyWheel'
        # jointIndex: 6 jointName: 'frame_to_gyros'
        self.handlebar_joint = 0
        self.front_wheel_joint = 2
        self.back_wheel_joint = 3
        self.fly_wheel_joint = 5
        self.gyros_link = 6
        self.MAX_FORCE = 2000

        self.distance_array_buffer = []
        self.max_buffer_size = 20
        self.number_of_frames = 3
        self.num_rays = 800
        self.ray_len = 100
        self.lidar_origin_offset = [0., 0., .7]  # 激光雷达相对于小车的位置偏移量
        self.initial_joint_positions = None
        self.initial_joint_velocities = None
        self.initial_position, self.initial_orientation = p.getBasePositionAndOrientation(self.bicycleId, self.client)

        # 设置飞轮速度上限
        p.changeDynamics(self.bicycleId,
                         self.fly_wheel_joint,
                         maxJointVelocity=max_flywheel_vel,
                         physicsClientId=self.client)

        # 获取自行车的刚体数量
        # num_joints = p.getNumJoints(self.bicycleId)
        # print("Number of joints: ", num_joints)
        # for i in range(num_joints):
        #     # 获取自行车每个部分的ID
        #     joint_info = p.getJointInfo(self.bicycleId, i, self.client)
        #     print("jointIndex: ", joint_info[0], "jointName: ", joint_info[1])
        self.obstacle_ids = obstacle_ids

    def apply_action(self, action):
        """
        Apply the action to the bicycle.

        Parameters:
        action[0]控制车把位置
        action[1]控制前后轮速度
        action[2]控制飞轮
        """
        # action[0] = frame_to_handlebar 车把位置控制
        p.setJointMotorControl2(bodyUniqueId=self.bicycleId,
                                jointIndex=self.handlebar_joint,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=action[0],
                                force=self.MAX_FORCE,
                                physicsClientId=self.client)

        # action[1] = handlebar_to_frontwheel 前轮速度控制
        p.setJointMotorControl2(bodyUniqueId=self.bicycleId,
                                jointIndex=self.front_wheel_joint,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=action[1],
                                force=self.MAX_FORCE,
                                physicsClientId=self.client)

        # action[1] = frame_to_backwheel 后轮速度控制
        p.setJointMotorControl2(bodyUniqueId=self.bicycleId,
                                jointIndex=self.back_wheel_joint,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=action[1],
                                force=self.MAX_FORCE,
                                physicsClientId=self.client)

        # action[2] = flyWheelLink_to_flyWheel 飞轮控制
        p.setJointMotorControl2(bodyUniqueId=self.bicycleId,
                                jointIndex=self.fly_wheel_joint,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=action[2],
                                force=self.MAX_FORCE,
                                physicsClientId=self.client)

    def get_observation(self):
        # Get the position位置 and orientation方向(姿态) of the bicycle in the simulation
        pos, ori = p.getBasePositionAndOrientation(bodyUniqueId=self.bicycleId, physicsClientId=self.client)
        # The rotation order is first roll around X, then pitch around Y and finally yaw around Z
        # p.getBaseVelocity()返回的格式 (线速度(x, y, z), 角速度(wx, wy, wz))
        # _, angular_velocity = p.getBaseVelocity(self.bicycleId, self.client)

        gyros_link_state = p.getLinkState(self.bicycleId, self.gyros_link, computeLinkVelocity=1,
                                          physicsClientId=self.client)
        gyros_link_orientation = gyros_link_state[1]
        link_ang = p.getEulerFromQuaternion(gyros_link_orientation)
        roll_angle = link_ang[0]
        yaw_angle = link_ang[2]
        gyros_link_angular_vel = gyros_link_state[7]
        # roll_angle_vel = gyros_link_angular_vel[0]

        handlebar_joint_state = p.getJointState(self.bicycleId, self.handlebar_joint, self.client)
        handlebar_joint_ang = handlebar_joint_state[0]
        # handlebar_joint_vel = handlebar_joint_state[1]

        back_wheel_joint_state = p.getJointState(self.bicycleId, self.back_wheel_joint, self.client)
        back_wheel_joint_vel = back_wheel_joint_state[1]

        fly_wheel_joint_state = p.getJointState(self.bicycleId, self.fly_wheel_joint, self.client)
        # fly_wheel_joint_ang = fly_wheel_joint_state[0] % (2 * math.pi)
        fly_wheel_joint_vel = fly_wheel_joint_state[1]

        lidar_info = self._get_lidar_info2(pos)
        is_collided = self._is_collided()

        # observation = [pos[0], pos[1],
        #                yaw_angle,
        #                roll_angle, roll_angle_vel,
        #                handlebar_joint_ang, handlebar_joint_vel,
        #                back_wheel_joint_vel, fly_wheel_joint_vel,
        #                lidar_info, is_collided
        #                ]

        observation = [pos[0],
                       pos[1],
                       yaw_angle,
                       roll_angle,
                       handlebar_joint_ang,
                       back_wheel_joint_vel,
                       fly_wheel_joint_vel,
                       lidar_info,
                       is_collided
                       ]

        return observation

    def reset(self):
        p.resetBasePositionAndOrientation(self.bicycleId, self.initial_position,
                                          self.initial_orientation, self.client)
        p.resetJointState(self.bicycleId, self.handlebar_joint, 0, 0, self.client)
        p.resetJointState(self.bicycleId, self.fly_wheel_joint, 0, 0, self.client)
        p.resetJointState(self.bicycleId, self.front_wheel_joint, 0, 0, self.client)
        p.resetJointState(self.bicycleId, self.back_wheel_joint, 0, 0, self.client)
        # self.image_stack = []  # 图像清空
        return self.get_observation()

    """
    def _get_lidar_info(self, bicycle_pos):
        if len(self.image_stack) == self.number_of_frames:
            self.image_stack.pop(0)

        while len(self.image_stack) < self.number_of_frames:
            rayFrom = []
            rayTo = []

            for i in range(self.num_rays):
                rayFrom.append(
                    [bicycle_pos[0] + self.lidar_origin_offset[0],
                     bicycle_pos[1] + self.lidar_origin_offset[1],
                     bicycle_pos[2] + self.lidar_origin_offset[2]])
                rayTo.append([
                    rayFrom[i][0] + self.ray_len * math.sin(2. * math.pi * float(i) / self.num_rays),
                    rayFrom[i][1] + self.ray_len * math.cos(2. * math.pi * float(i) / self.num_rays),
                    rayFrom[i][2]
                ])

            results = p.rayTestBatch(rayFromPositions=rayFrom, rayToPositions=rayTo)
            radar_image = _map_to_image(results)
            radar_image = np.expand_dims(radar_image, axis=0)  # 增加通道维度，使形状变为 (1, H, W)
            self.image_stack.append(radar_image)  # 将当前图像添加到列表中

        radar_images = np.concatenate(self.image_stack, axis=0)

        return radar_images
    """

    def _is_collided(self):
        for obstacle_id in self.obstacle_ids:
            contact_points = p.getClosestPoints(self.bicycleId, obstacle_id, distance=0.0, physicsClientId=self.client)
            if len(contact_points) > 0:
                return True
        return False

    def _get_lidar_info2(self, bicycle_pos):
        rayFrom, rayTo = [], []
        ray_start_x = bicycle_pos[0] + self.lidar_origin_offset[0]
        ray_start_y = bicycle_pos[1] + self.lidar_origin_offset[1]
        ray_start_z = bicycle_pos[2] + self.lidar_origin_offset[2]

        for i in range(self.num_rays):
            rayFrom.append([ray_start_x, ray_start_y, ray_start_z])
            rayTo.append([
                rayFrom[i][0] + self.ray_len * math.sin(2. * math.pi * float(i) / self.num_rays),
                rayFrom[i][1] + self.ray_len * math.cos(2. * math.pi * float(i) / self.num_rays),
                rayFrom[i][2]])

        results = p.rayTestBatch(rayFromPositions=rayFrom, rayToPositions=rayTo)
        distance = []
        for res in results:
            if res[0] < 0:
                distance.append(self.ray_len)
            else:
                distance.append(np.linalg.norm(np.array(res[3]) - np.array([ray_start_x, ray_start_y, ray_start_z])))

        # for i, result in enumerate(results):
        #     if result[0] < 0:
        #         p.addUserDebugLine(rayFrom[i], rayTo[i], lineColorRGB=[0, 1, 0], lineWidth=1.0)
        #     else:
        #         hit_position = result[3]
        #         # 计算击中点到射线起点的距离
        #         distance = ((hit_position[0] - rayFrom[i][0]) ** 2 +
        #                     (hit_position[1] - rayFrom[i][1]) ** 2 +
        #                     (hit_position[2] - rayFrom[i][2]) ** 2) ** 0.5
        #         # 在击中点附近显示距离
        #         text_position = (hit_position[0], hit_position[1], hit_position[2] + 0.1)  # 提高文本显示位置
        #         p.addUserDebugText(f"{distance:.2f} m", text_position, textColorRGB=[1, 1, 1], textSize=1.0)
        #         # 显示击中点的射线
        #         p.addUserDebugLine(rayFrom[i], hit_position, lineColorRGB=[1, 0, 0], lineWidth=1.0)

        return np.array(distance, dtype=np.float32)

    def _get_lidar_info3(self, bicycle_pos, bicycle_ori):
        # Convert quaternion orientation to Euler angles (yaw, pitch, roll)
        ang = p.getEulerFromQuaternion(bicycle_ori)
        yaw = ang[2]

        # Only scan the front 180 degrees (from -90° to +90° of yaw)
        rayFrom, rayTo = [], []
        for i in range(self.num_rays):
            # 计算每个射线的角度，射线从yaw-90°开始到yaw+90°结束
            angle = yaw - math.pi / 2 + (math.pi * float(i) / (self.num_rays - 1))  # 角度范围从yaw-90°到yaw+90°

            # Calculate the starting position of the ray (lidar origin offset)
            rayFrom.append([
                bicycle_pos[0] + self.lidar_origin_offset[0],
                bicycle_pos[1] + self.lidar_origin_offset[1],
                bicycle_pos[2] + self.lidar_origin_offset[2]
            ])

            # Calculate the direction of the ray (180° sweep)
            rayTo.append([
                rayFrom[i][0] + self.ray_len * math.cos(angle),
                rayFrom[i][1] + self.ray_len * math.sin(angle),
                rayFrom[i][2]
            ])

        results = p.rayTestBatch(rayFromPositions=rayFrom, rayToPositions=rayTo)

        distance = [
            np.linalg.norm(np.array([result[3][0], result[3][1]]) - np.array([bicycle_pos[0], bicycle_pos[1]]))
            for result in results
        ]

        return np.array(distance, dtype=np.float32)
