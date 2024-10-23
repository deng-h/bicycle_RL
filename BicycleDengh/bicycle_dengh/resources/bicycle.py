import pybullet as p
import random
import math
import os
import platform
import numpy as np
import gymnasium as gym


class Bicycle:
    def __init__(self, client):
        self.client = client

        system = platform.system()
        if system == "Windows":
            f_name = os.path.join(os.path.dirname(__file__), 'bicycle_urdf\\bike.xml')
        else:
            f_name = os.path.join(os.path.dirname(__file__), 'bicycle_urdf/bike.xml')

        startOrientation = p.getQuaternionFromEuler([0, 0, 1.57])
        self.bicycleId = p.loadURDF(fileName=f_name, basePosition=[0, 0, 1], baseOrientation=startOrientation)
        self.handlebar_joint = 0
        self.front_wheel_joint = 1
        self.back_wheel_joint = 2
        self.fly_wheel_joint = 4
        self.gyros_link = 5
        self.MAX_FORCE = 2000
        self.max_flywheel_vel = 120.0  # 飞轮速度上限

        self.initial_joint_positions = None
        self.initial_joint_velocities = None
        self.initial_position, self.initial_orientation = p.getBasePositionAndOrientation(self.bicycleId)

    def apply_action(self, action):
        """
        Apply the action to the bicycle.
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
        # Get the position and orientation of the bicycle in the simulation
        pos, _ = p.getBasePositionAndOrientation(self.bicycleId, self.client)
        # The rotation order is first roll around X, then pitch around Y and finally yaw around Z
        # euler_angles = p.getEulerFromQuaternion(orn)  # 将四元数转换为欧拉角
        # yaw = euler_angles[2]  # 获取偏航角（yaw）
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

        observation = [pos[0], pos[1], yaw_angle,
                       roll_angle, roll_angular_vel,
                       handlebar_joint_ang, handlebar_joint_vel,
                       back_wheel_joint_vel, fly_wheel_joint_vel
                       ]

        return observation

    def get_action_space(self):
        """
        返回动作空间
        :return: [车把角度，前后轮速度, 飞轮速度]
        """
        return gym.spaces.box.Box(
            low=np.array([-1.57, 0.0, -self.max_flywheel_vel]),
            high=np.array([1.57, 5.0, self.max_flywheel_vel]),
            shape=(3,),
            dtype=np.float32)

    def get_observation_space(self):
        """
        返回观测空间
        :return: [机器人与目标点距离, 机器人与目标点的角度, 翻滚角, 翻滚角角速度, 车把角度, 车把角速度, 后轮速度, 飞轮速度]
        """
        return gym.spaces.box.Box(
            low=np.array([0.0, -math.pi, -math.pi, -15.0, -1.57, -15.0, 0.0, -self.max_flywheel_vel]),
            high=np.array([100.0, math.pi, math.pi, 15.0, 1.57, 15.0, 10.0, self.max_flywheel_vel]),
            shape=(8,),
            dtype=np.float32)

    def reset(self):
        p.resetBasePositionAndOrientation(self.bicycleId, self.initial_position, self.initial_orientation)
        p.resetJointState(self.bicycleId, self.handlebar_joint, targetValue=0, targetVelocity=0)
        p.resetJointState(self.bicycleId, self.fly_wheel_joint, targetValue=0, targetVelocity=0)
        p.resetJointState(self.bicycleId, self.front_wheel_joint, targetValue=0, targetVelocity=0)
        p.resetJointState(self.bicycleId, self.back_wheel_joint, targetValue=0, targetVelocity=0)
        return self.get_observation()
