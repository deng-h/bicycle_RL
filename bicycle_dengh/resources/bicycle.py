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

        startOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.bicycleId = p.loadURDF(fileName=f_name, basePosition=[0, 0, 1], baseOrientation=startOrientation)

        self.handlebar_joint = 0
        self.front_wheel_joint = 1
        self.back_wheel_joint = 2
        self.fly_wheel_joint = 4
        self.gyros_link = 5
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
        # p.getBaseVelocity()返回的格式 (线速度(x, y, z), 角速度(wx, wy, wz))
        # _, angular_velocity = p.getBaseVelocity(self.bicycleId, self.client)

        gyros_link_state = p.getLinkState(self.bicycleId, self.gyros_link, computeLinkVelocity=1)
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
