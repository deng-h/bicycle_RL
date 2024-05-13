import pybullet as p
import random
import math
import os


class Bicycle:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'bicycle_urdf\\bike.xml')
        self.bicycleId = p.loadURDF(fileName=f_name,
                                    basePosition=[0, 0, 1])
        self.handlebar_joint = 0
        self.front_wheel_joint = 1
        self.back_wheel_joint = 2
        self.fly_wheel_joint = 4
        self.MAX_FORCE = 500
        self.handlebar_angle = 0
        # 30%的概率触发扰动
        self.noise_probability = 0.3

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
        self.handlebar_angle = action[0]

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
                                controlMode=p.TORQUE_CONTROL,
                                force=action[2],
                                physicsClientId=self.client)

        # 产生随机扰动
        # random_number = random.random()
        # if random_number < self.noise_probability:
        #     force_magnitude = 30
        #     p.applyExternalForce(objectUniqueId=self.bicycle,
        #                          linkIndex=-1,  # link index or -1 for the base
        #                          forceObj=[random.uniform(-force_magnitude, force_magnitude),
        #                                    random.uniform(-force_magnitude, force_magnitude), 0],
        #                          posObj=[0, 0, 0],
        #                          flags=p.LINK_FRAME)

    def get_observation(self):
        """ 
        Returns:
        (位置x, 位置y, 翻滚角roll, 车把角度)
        """
        # Get the position位置 and orientation方向(姿态) of the bicycle in the simulation
        pos, ang = p.getBasePositionAndOrientation(self.bicycleId, self.client)
        # Convert the orientation to euler angles
        # 欧拉角 The rotation order is first roll around X, then pitch around Y and finally yaw around Z
        ang = p.getEulerFromQuaternion(ang)

        # p.getBaseVelocity()返回的格式 (线速度(x, y, z), 角速度(wx, wy, wz)) 这里只取线速度的x, y
        vel_xy = p.getBaseVelocity(self.bicycleId, self.client)[0][0:2]
        vel_x = vel_xy[0]
        vel_y = vel_xy[1]
        vel = math.sqrt(vel_x ** 2 + vel_y ** 2)

        observation = (pos[0], pos[1], ang[0], vel, self.handlebar_angle)

        return observation
