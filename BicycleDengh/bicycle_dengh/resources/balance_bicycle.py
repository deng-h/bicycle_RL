import pybullet as p
import random
import math
import os


class BalanceBicycle:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'bicycle_urdf\\bike.xml')
        self.bicycleId = p.loadURDF(fileName=f_name, basePosition=[0, 0, 1])
        self.handlebar_joint = 0
        self.front_wheel_joint = 1
        self.back_wheel_joint = 2
        self.fly_wheel_joint = 4
        self.MAX_FORCE = 2000
        # 30%的概率触发扰动
        # self.noise_probability = 0.3

    def apply_action(self, action):
        """
        Apply the action to the bicycle.

        Parameters:
        action[0]控制飞轮
        """
        p.setJointMotorControl2(bodyUniqueId=self.bicycleId,
                                jointIndex=self.fly_wheel_joint,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=action[0],
                                force=self.MAX_FORCE,
                                physicsClientId=self.client)

        # p.setJointMotorControl2(bodyUniqueId=self.bicycleId,
        #                         jointIndex=self.fly_wheel_joint,
        #                         controlMode=p.TORQUE_CONTROL,
        #                         force=action[0],
        #                         physicsClientId=self.client)

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
        [roll_angle, roll_vel, fly_wheel_joint_ang, fly_wheel_joint_vel]
        """
        # Get the position位置 and orientation方向(姿态) of the bicycle in the simulation
        _, ang = p.getBasePositionAndOrientation(self.bicycleId, self.client)
        # The rotation order is first roll around X, then pitch around Y and finally yaw around Z
        ang = p.getEulerFromQuaternion(ang)
        roll_angle = ang[0]
        # p.getBaseVelocity()返回的格式 (线速度(x, y, z), 角速度(wx, wy, wz))
        _, angular_velocity = p.getBaseVelocity(self.bicycleId, self.client)
        roll_vel = angular_velocity[0]

        fly_wheel_joint_state = p.getJointState(self.bicycleId, self.fly_wheel_joint, self.client)
        fly_wheel_joint_ang = fly_wheel_joint_state[0] % (2 * math.pi)
        fly_wheel_joint_vel = fly_wheel_joint_state[1]

        observation = [roll_angle, roll_vel, fly_wheel_joint_ang, fly_wheel_joint_vel]

        return observation
