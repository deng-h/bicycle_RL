import pybullet as p
import random
import math
import os
import numpy as np
import platform
import matplotlib.pyplot as plt


class BicycleCamera:
    def __init__(self, client, max_flywheel_vel):
        self.client = client

        system = platform.system()
        if system == "Windows":
            f_name = os.path.join(os.path.dirname(__file__), 'bicycle_urdf\\bike.xml')
        else:
            f_name = os.path.join(os.path.dirname(__file__), 'bicycle_urdf/bike.xml')

        startOrientation = p.getQuaternionFromEuler([0, 0, 1.57])
        self.bicycleId = p.loadURDF(fileName=f_name, basePosition=[0, 0, 1], baseOrientation=startOrientation,
                                    physicsClientId=self.client)
        self.handlebar_joint = 0
        self.camera_joint = 1
        self.front_wheel_joint = 2
        self.back_wheel_joint = 3
        self.fly_wheel_joint = 5
        self.gyros_link = 6
        self.MAX_FORCE = 2000

        self.pitch_angle_rad = np.radians(-20.0)  # 定义相机的俯仰角（正值为向上俯仰，负值为向下俯仰）
        self.camera_offset_distance = 0.05  # 定义需要将相机视角位置移出的距离，使相机移到长方体外部
        self.camera_target_distance = 0.7  # 定义目标位置距离相机的前进距离，也就是定义相机的视野距离
        self.local_camera_direction = [1, 0, 0]  # 相机前方向向量（局部坐标系中）[1, 0, 0] 表示相机的前方
        # 在局部坐标系中通过俯仰角度调整相机前方向的y、z分量
        self.local_pitched_direction = [np.cos(self.pitch_angle_rad), 0, np.sin(self.pitch_angle_rad)]
        self.image_stack = []  # 图像列表
        self.number_of_frames = 3
        self.image_width = 128
        self.image_height = 128

        # projectionMatrix定义了如何将三维场景投影到二维图像上，包括视野、长宽比和远近裁剪平面。可以理解为“拍摄效果的配置”
        self.projectionMatrix = p.computeProjectionMatrixFOV(
            fov=60.0,  # 视野角度，角度越大视野越宽，但失真可能越明显
            aspect=1.0,  # 图像的宽高比，例如 640/480 或 1.0，确保图像不被拉伸或压缩
            nearVal=0.1,  # nearVal 和 farVal 决定了渲染图像的范围 远近裁剪平面通常分别设置为 0.1 和 100，确保在视图中显示足够的景物而不出现异常裁剪
            farVal=100.0)

        self.initial_joint_positions = None
        self.initial_joint_velocities = None
        self.initial_position, self.initial_orientation = p.getBasePositionAndOrientation(self.bicycleId, self.client)

        # 设置飞轮速度上限
        p.changeDynamics(self.bicycleId,
                         self.fly_wheel_joint,
                         maxJointVelocity=max_flywheel_vel,
                         physicsClientId=self.client)

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
        pos, _ = p.getBasePositionAndOrientation(bodyUniqueId=self.bicycleId, physicsClientId=self.client)
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
        roll_angular_vel = gyros_link_angular_vel[0]

        handlebar_joint_state = p.getJointState(self.bicycleId, self.handlebar_joint, self.client)
        handlebar_joint_ang = handlebar_joint_state[0]
        handlebar_joint_vel = handlebar_joint_state[1]

        back_wheel_joint_state = p.getJointState(self.bicycleId, self.back_wheel_joint, self.client)
        back_wheel_joint_vel = back_wheel_joint_state[1]

        fly_wheel_joint_state = p.getJointState(self.bicycleId, self.fly_wheel_joint, self.client)
        # fly_wheel_joint_ang = fly_wheel_joint_state[0] % (2 * math.pi)
        fly_wheel_joint_vel = fly_wheel_joint_state[1]

        depth_images = self._get_image()
        if depth_images.shape != (3, 128, 128):
            print(depth_images.shape)

        observation = [pos[0], pos[1], yaw_angle,
                       roll_angle, roll_angular_vel,
                       handlebar_joint_ang, handlebar_joint_vel,
                       back_wheel_joint_vel, fly_wheel_joint_vel,
                       depth_images
                       ]

        return observation

    def reset(self):
        p.resetBasePositionAndOrientation(self.bicycleId, self.initial_position,
                                          self.initial_orientation, self.client)
        p.resetJointState(self.bicycleId, self.handlebar_joint, 0, 0, self.client)
        p.resetJointState(self.bicycleId, self.fly_wheel_joint, 0, 0, self.client)
        p.resetJointState(self.bicycleId, self.front_wheel_joint, 0, 0, self.client)
        p.resetJointState(self.bicycleId, self.back_wheel_joint, 0, 0, self.client)
        self.image_stack = []  # 图像清空
        return self.get_observation()

    def _get_image(self):
        camera_state = p.getLinkState(self.bicycleId, self.camera_joint, physicsClientId=self.client)
        camera_position = camera_state[0]  # 相机的位置
        camera_orientation = camera_state[1]  # 包含四元数（x, y, z, w）
        # 使用四元数将局部方向转换到世界坐标系中 将方向向量转换为世界坐标系
        world_camera_direction = p.rotateVector(camera_orientation, self.local_camera_direction)
        # 将俯仰调整后的前方向量转换为世界坐标系，用于计算 target_position，从而让相机在俯仰角度调整的视线方向上拍摄
        world_pitched_direction = p.rotateVector(camera_orientation, self.local_pitched_direction)

        # 计算相机在世界坐标系中的目标位置
        target_position = [
            camera_position[0] + self.camera_target_distance * world_pitched_direction[0],
            camera_position[1] + self.camera_target_distance * world_pitched_direction[1],
            camera_position[2] + self.camera_target_distance * world_pitched_direction[2]
        ]

        # 计算新的相机位置，使相机在长方体前方
        cameraEyePosition = [
            camera_position[0] + self.camera_offset_distance * world_camera_direction[0],
            camera_position[1] + self.camera_offset_distance * world_camera_direction[1],
            camera_position[2] + self.camera_offset_distance * world_camera_direction[2]
        ]

        # 获取视图矩阵
        viewMatrix = p.computeViewMatrix(
            cameraEyePosition=cameraEyePosition,  # 相机的实际位置，例如 [x, y, z] 坐标
            cameraTargetPosition=target_position,  # 相机所看的目标点位置，例如设置在相机前方的一点，通常与相机的前进方向一致
            cameraUpVector=[0, 0, 1])  # 决定相机的“上”方向，例如 [0, 0, 1] 表示 z 轴为上。若要倾斜相机可以更改该向量

        if len(self.image_stack) == self.number_of_frames:
            self.image_stack.pop(0)

        # 获取并渲染相机画面
        # DIRECT mode does allow rendering of images using the built-in software renderer
        # through the 'getCameraImage' API. 也就是说开DIRECT模式也能获取图像
        # getCameraImage 将返回一幅 RGB 图像、一个深度缓冲区和一个分割掩码缓冲区，其中每个像素都有可见物体的唯一 ID
        while len(self.image_stack) < self.number_of_frames:
            _, _, _, depth_img, _ = p.getCameraImage(
                width=self.image_width,
                height=self.image_height,
                viewMatrix=viewMatrix,
                projectionMatrix=self.projectionMatrix,
                physicsClientId=self.client,
            )
            depth_img = np.expand_dims(depth_img, axis=0)  # 增加通道维度，使形状变为 (1, H, W)
            self.image_stack.append(depth_img)  # 将当前图像添加到列表中

        depth_images = np.concatenate(self.image_stack, axis=0)

        # depth_map = np.array(depth_img)
        # plt.imshow(depth_map)
        # plt.savefig('./depth_map_matplotlib.png')

        return depth_images
