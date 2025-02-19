import pybullet as p
import math
import os
import numpy as np
import platform


class ZBicycle:
    def __init__(self, client, max_flywheel_vel=120.0, obstacle_ids=None):
        self.client = client

        system = platform.system()
        if system == "Windows":
            f_name = os.path.join(os.path.dirname(__file__), 'bicycle_urdf\\bike.xml')
        else:
            f_name = os.path.join(os.path.dirname(__file__), 'bicycle_urdf/bike.xml')

        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.bicycleId = p.loadURDF(fileName=f_name, basePosition=[1, 1, 1], baseOrientation=startOrientation,
                                    physicsClientId=self.client)
        # Number of joints: 7
        # jointIndex: 0 jointName: 'frame_to_handlebar'
        # jointIndex: 已删除 jointName: 'camera_joint'
        # jointIndex: 1 jointName: 'handlebar_to_frontwheel'
        # jointIndex: 2 jointName: 'frame_to_backwheel'
        # jointIndex: 3 jointName: 'frame_to_flyWheelLink'
        # jointIndex: 4 jointName: 'flyWheelLink_to_flyWheel'
        # jointIndex: 5 jointName: 'frame_to_gyros'
        self.handlebar_joint = 0
        self.front_wheel_joint = 1
        self.back_wheel_joint = 2
        self.fly_wheel_joint = 4
        self.gyros_link = 5
        self.MAX_FORCE = 2000

        self.max_buffer_size = 20
        self.number_of_frames = 3
        self.num_rays = 120
        self.ray_len = 50
        self.lidar_origin_offset = [0., 0., .7]  # 激光雷达相对于小车的位置偏移量
        self.initial_joint_positions = None
        self.initial_joint_velocities = None
        self.initial_position, self.initial_orientation = p.getBasePositionAndOrientation(self.bicycleId, self.client)

        self.lidar_update_frequency = 3  # 每 n 帧更新一次激光雷达
        self.collision_check_frequency = 3  # 每 n 帧检查一次碰撞
        self.frame_count = 0
        # 初始化 last_lidar_info 用于在不更新激光雷达信息的帧中，仍然提供一个值，避免程序出错
        self.last_lidar_info = np.full(self.num_rays, self.ray_len, dtype=np.float32)

        # 获取自行车的刚体数量
        # num_joints = p.getNumJoints(self.bicycleId)
        # print("Number of joints: ", num_joints)
        # for i in range(num_joints):
        #     # 获取自行车每个部分的ID
        #     joint_info = p.getJointInfo(self.bicycleId, i, self.client)
        #     print("jointIndex: ", joint_info[0], "jointName: ", joint_info[1])
        self.obstacle_ids = obstacle_ids
        self.sin_array = [self.ray_len * math.sin(2. * math.pi * float(i) / self.num_rays) for i in
                          range(self.num_rays)]
        self.cos_array = [self.ray_len * math.cos(2. * math.pi * float(i) / self.num_rays) for i in
                          range(self.num_rays)]

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

    def apply_action2(self, RL_action, PID_action):
        """
        Apply the action to the bicycle.控制分为两部分，前后轮速度和车把位置是RL控制，飞轮是PID控制。

        Parameters:
        RL_action[0]控制车把位置
        RL_action[1]控制前后轮速度
        PID_action[0]控制飞轮
        """
        # action[0] = frame_to_handlebar 车把位置控制
        p.setJointMotorControl2(bodyUniqueId=self.bicycleId,
                                jointIndex=self.handlebar_joint,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=RL_action[0],
                                force=self.MAX_FORCE,
                                physicsClientId=self.client)

        # action[1] = handlebar_to_frontwheel 前轮速度控制
        p.setJointMotorControl2(bodyUniqueId=self.bicycleId,
                                jointIndex=self.front_wheel_joint,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=RL_action[1],
                                force=self.MAX_FORCE,
                                physicsClientId=self.client)

        # action[1] = frame_to_backwheel 后轮速度控制
        p.setJointMotorControl2(bodyUniqueId=self.bicycleId,
                                jointIndex=self.back_wheel_joint,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=RL_action[1],
                                force=self.MAX_FORCE,
                                physicsClientId=self.client)

        # action[2] = flyWheelLink_to_flyWheel 飞轮控制
        p.setJointMotorControl2(bodyUniqueId=self.bicycleId,
                                jointIndex=self.fly_wheel_joint,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=PID_action,
                                force=self.MAX_FORCE,
                                physicsClientId=self.client)

    def get_observation(self):
        self.frame_count += 1
        # Get the position位置 and orientation方向(姿态) of the bicycle in the simulation
        # pos, _ = p.getBasePositionAndOrientation(bodyUniqueId=self.bicycleId, physicsClientId=self.client)
        # The rotation order is first roll around X, then pitch around Y and finally yaw around Z
        # p.getBaseVelocity()返回的格式 (线速度(x, y, z), 角速度(wx, wy, wz))
        # _, angular_velocity = p.getBaseVelocity(self.bicycleId, self.client)

        gyros_link_state = p.getLinkState(self.bicycleId, self.gyros_link, computeLinkVelocity=1,
                                          physicsClientId=self.client)
        gyros_link_orientation = gyros_link_state[1]
        link_ang = p.getEulerFromQuaternion(gyros_link_orientation)
        roll_angle = link_ang[0]
        # yaw_angle = link_ang[2]
        gyros_link_angular_vel = gyros_link_state[7]
        roll_angle_vel = gyros_link_angular_vel[0]

        # handlebar_joint_state = p.getJointState(self.bicycleId, self.handlebar_joint, self.client)
        # handlebar_joint_ang = handlebar_joint_state[0]
        # handlebar_joint_vel = handlebar_joint_state[1]

        # back_wheel_joint_state = p.getJointState(self.bicycleId, self.back_wheel_joint, self.client)
        # back_wheel_joint_vel = back_wheel_joint_state[1]

        fly_wheel_joint_state = p.getJointState(self.bicycleId, self.fly_wheel_joint, self.client)
        # fly_wheel_joint_ang = fly_wheel_joint_state[0] % (2 * math.pi)
        fly_wheel_joint_vel = fly_wheel_joint_state[1]

        # lidar_info = None  # 初始化 lidar_info
        # if self.frame_count % self.lidar_update_frequency == 0:
        #     lidar_info = self._get_lidar_info3(pos)

        # is_collided = self._is_collided()

        # observation = [pos[0],
        #                pos[1],
        #                yaw_angle,
        #                roll_angle,
        #                roll_angle_vel,
        #                handlebar_joint_ang,
        #                handlebar_joint_vel,
        #                back_wheel_joint_vel,
        #                fly_wheel_joint_vel,
        #                lidar_info if lidar_info is not None else self.last_lidar_info,  # 使用上次的激光雷达信息
        #                ]
        # 保存激光雷达信息，下次使用
        # self.last_lidar_info = lidar_info if lidar_info is not None else self.last_lidar_info

        observation = [roll_angle,
                       roll_angle_vel,
                       fly_wheel_joint_vel,
                       ]

        return observation

    def reset(self):
        p.resetBasePositionAndOrientation(self.bicycleId, self.initial_position,
                                          self.initial_orientation, self.client)
        p.resetJointState(self.bicycleId, self.handlebar_joint, 0, 0, self.client)
        p.resetJointState(self.bicycleId, self.fly_wheel_joint, 0, 0, self.client)
        p.resetJointState(self.bicycleId, self.front_wheel_joint, 0, 0, self.client)
        p.resetJointState(self.bicycleId, self.back_wheel_joint, 0, 0, self.client)
        # self.last_lidar_info = np.full(self.num_rays, self.ray_len, dtype=np.float32)  # 初始化 last_lidar_info
        # self.frame_count = 0
        return self.get_observation()

    def _is_collided(self):
        for obstacle_id in self.obstacle_ids:
            contact_points = p.getClosestPoints(self.bicycleId, obstacle_id, distance=0.2, physicsClientId=self.client)
            if len(contact_points) > 0:
                return True
        return False

    def _get_lidar_info3(self, bicycle_pos):
        # 计算射线起点的坐标
        ray_start = np.array(bicycle_pos) + np.array(self.lidar_origin_offset)

        # 创建射线起点和终点的数组
        rayFrom = np.tile(ray_start, (self.num_rays, 1))
        rayTo = rayFrom + np.column_stack((self.sin_array, self.cos_array, np.zeros(self.num_rays)))

        results = p.rayTestBatch(rayFromPositions=rayFrom.tolist(), rayToPositions=rayTo.tolist())

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

        # 计算距离
        distance = np.array([
            self.ray_len if res[0] < 0 else np.linalg.norm(np.array(res[3]) - ray_start)
            for res in results
        ], dtype=np.float32)

        return distance

    def draw_circle(self, center_pos, radius, num_segments=24, color=None):
        """
        在PyBullet中绘制圆形。

        Args:
            center_pos (list or np.array): 圆心位置 [x, y, z]。
            radius (float): 圆的半径。
            num_segments (int): 用于近似圆形的线段数量。默认值为24。
            color (list): 圆形的颜色，RGB格式，例如 [1, 0, 0] 代表红色。
        """
        if color is None:
            color = [1, 0, 0]
        points = []
        for i in range(num_segments):
            angle = 2 * np.pi * i / num_segments
            x = center_pos[0] + radius * np.cos(angle)
            y = center_pos[1] + radius * np.sin(angle)
            z = center_pos[2]  # 保持与圆心相同的z坐标
            points.append([x, y, z])

        # 绘制线段连接点，形成圆形
        for i in range(num_segments):
            p.addUserDebugLine(
                lineFromXYZ=points[i],
                lineToXYZ=points[(i + 1) % num_segments],  # 连接到下一个点，最后一个点连接到第一个点
                lineColorRGB=color
            )
