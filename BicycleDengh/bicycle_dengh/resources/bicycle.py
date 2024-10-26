import pybullet as p
import random
import math
import os
import platform


class Bicycle:
    def __init__(self, client, max_flywheel_vel):
        self.client = client

        system = platform.system()
        if system == "Windows":
            f_name = os.path.join(os.path.dirname(__file__), 'bicycle_urdf\\bike.xml')
        else:
            f_name = os.path.join(os.path.dirname(__file__), 'bicycle_urdf/bike.xml')

        startOrientation = p.getQuaternionFromEuler([0, 0, 1.57])
        self.bicycleId = p.loadURDF(fileName=f_name, basePosition=[0, 0, 1], baseOrientation=startOrientation)
        self.handlebar_joint = 0
        self.camera_joint = 1
        self.front_wheel_joint = 2
        self.back_wheel_joint = 3
        self.fly_wheel_joint = 5
        self.gyros_link = 6
        self.MAX_FORCE = 2000

        self.initial_joint_positions = None
        self.initial_joint_velocities = None
        self.initial_position, self.initial_orientation = p.getBasePositionAndOrientation(self.bicycleId)

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

        camera_state = p.getLinkState(self.bicycleId, self.camera_joint)
        camera_position = camera_state[0]  # 相机的位置
        camera_orientation = camera_state[1]  # 相机的四元数方向
        # 使用相机方向作为目标方向，使相机视角保持前向
        target_position = [
            camera_position[0] + camera_orientation[0],
            camera_position[1] + camera_orientation[1],
            camera_position[2] + camera_orientation[2]
        ]

        # 获取视图矩阵
        viewMatrix = p.computeViewMatrix(
            cameraEyePosition=camera_position,  # 相机的实际位置，例如 [x, y, z] 坐标
            cameraTargetPosition=target_position,  # 相机所看的目标点位置，例如设置在相机前方的一点，通常与相机的前进方向一致
            cameraUpVector=[0, 0, 1])  # 决定相机的“上”方向，例如 [0, 0, 1] 表示 z 轴为上。若要倾斜相机可以更改该向量

        # projectionMatrix定义了如何将三维场景投影到二维图像上，包括视野、长宽比和远近裁剪平面。可以理解为“拍摄效果的配置”
        projectionMatrix = p.computeProjectionMatrixFOV(
            fov=60.0,  # 视野角度，角度越大视野越宽，但失真可能越明显
            aspect=1.0,  # 图像的宽高比，例如 640/480 或 1.0，确保图像不被拉伸或压缩
            nearVal=0.1,  # nearVal 和 farVal 决定了渲染图像的范围 远近裁剪平面通常分别设置为 0.1 和 100，确保在视图中显示足够的景物而不出现异常裁剪
            farVal=100.0)

        # 获取并渲染相机画面
        # DIRECT mode does allow rendering of images using the built-in software renderer
        # through the 'getCameraImage' API. 也就是说开DIRECT模式也能获取图像
        # getCameraImage 将返回一幅 RGB 图像、一个深度缓冲区和一个分割掩码缓冲区，其中每个像素都有可见物体的唯一 ID
        width, height, _, depth_img, _ = p.getCameraImage(
            width=320,
            height=240,
            viewMatrix=viewMatrix,
            projectionMatrix=projectionMatrix,
            physicsClientId=self.client,
            flags=p.ER_NO_SEGMENTATION_MASK,  # 不使用分割实例图
        )

        observation = [pos[0], pos[1], yaw_angle,
                       roll_angle, roll_angular_vel,
                       handlebar_joint_ang, handlebar_joint_vel,
                       back_wheel_joint_vel, fly_wheel_joint_vel
                       ]

        return observation

    def reset(self):
        p.resetBasePositionAndOrientation(self.bicycleId, self.initial_position, self.initial_orientation)
        p.resetJointState(self.bicycleId, self.handlebar_joint, targetValue=0, targetVelocity=0)
        p.resetJointState(self.bicycleId, self.fly_wheel_joint, targetValue=0, targetVelocity=0)
        p.resetJointState(self.bicycleId, self.front_wheel_joint, targetValue=0, targetVelocity=0)
        p.resetJointState(self.bicycleId, self.back_wheel_joint, targetValue=0, targetVelocity=0)
        return self.get_observation()
