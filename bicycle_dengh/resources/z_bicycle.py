import pybullet as p
import math
import os
import numpy as np
import platform
import random


class ZBicycle:
    def __init__(self, client, max_flywheel_vel=120.0, obstacle_ids=None):
        self.client = client

        system = platform.system()
        if system == "Windows":
            f_name = os.path.join(os.path.dirname(__file__), 'bicycle_urdf\\bike.xml')
        else:
            f_name = os.path.join(os.path.dirname(__file__), 'bicycle_urdf/bike.xml')

        startOrientation = p.getQuaternionFromEuler([0., 0., 0.])
        self.bicycleId = p.loadURDF(fileName=f_name, basePosition=[0., 0., 1], baseOrientation=startOrientation,
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

        # 产生随机扰动
        # random_number = random.random()
        # if random_number < 0.3:
        #     force_magnitude = 30
        #     p.applyExternalForce(objectUniqueId=self.bicycleId,
        #                          linkIndex=-1,  # link index or -1 for the base
        #                          forceObj=[0, random.uniform(-force_magnitude, force_magnitude), 0],
        #                          posObj=[-0.3, -0.0, 1.5],
        #                          flags=p.WORLD_FRAME,
        #                          physicsClientId=self.client
        #                          )


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

    def apply_lateral_disturbance(self, disturbance_force, disturbance_position):
        """
        Apply lateral disturbance to the bicycle.

        Parameters:
        disturbance_force (float): The force of the lateral disturbance.
        disturbance_position (list): The position of the lateral disturbance [x, y, z].
        """
        p.applyExternalForce(
            objectUniqueId=self.bicycleId,
            linkIndex=-1,
            forceObj=[0, disturbance_force, 0],
            posObj=disturbance_position,
            flags=p.WORLD_FRAME,
            physicsClientId=self.client
        )
