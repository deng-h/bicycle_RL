import pybullet as p
import random
import math
import os


# 将角度从度数转换为弧度
def degrees_to_radians(degrees):
    return degrees * (math.pi / 180.0)


class BalanceBicycle:
    def __init__(self, client):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'bicycle_urdf\\bike.xml')
        # f_name = os.path.join(os.path.dirname(__file__), 'bicycle_urdf/bike.xml')
        roll_radians = degrees_to_radians(0.0)
        euler_angles = [roll_radians, 0.0, 0.0]
        quaternion = p.getQuaternionFromEuler(euler_angles)
        self.bicycleId = p.loadURDF(fileName=f_name, basePosition=[0, 0, 1], baseOrientation=quaternion)
        self.handlebar_joint = 0
        self.front_wheel_joint = 1
        self.back_wheel_joint = 2
        self.frame_to_flywheel_joint = 3  # 车架到飞轮杆的连接
        self.fly_wheel_joint = 4
        self.gyros_link = 5
        self.MAX_FORCE = 2000
        # self.noise_probability = 0.3  # 30%的概率触发扰动

        self.initial_joint_positions = None
        self.initial_joint_velocities = None
        self.initial_position, self.initial_orientation = p.getBasePositionAndOrientation(self.bicycleId)

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
        #     p.applyExternalForce(objectUniqueId=self.bicycleId,
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
        # _, ang = p.getBasePositionAndOrientation(self.bicycleId, self.client)
        # The rotation order is first roll around X, then pitch around Y and finally yaw around Z
        # ang = p.getEulerFromQuaternion(ang)
        # roll_angle = ang[0]
        # p.getBaseVelocity()返回的格式 (线速度(x, y, z), 角速度(wx, wy, wz))
        # _, angular_velocity = p.getBaseVelocity(self.bicycleId, self.client)
        # roll_vel = angular_velocity[0]
        # 陀螺仪的状态
        gyros_link_state = p.getLinkState(self.bicycleId, self.gyros_link, computeLinkVelocity=1)
        gyros_link_orientation = gyros_link_state[1]
        link_ang = p.getEulerFromQuaternion(gyros_link_orientation)
        roll_angle = link_ang[0]
        gyros_link_angular_vel = gyros_link_state[7]
        roll_angular_vel = gyros_link_angular_vel[0]

        fly_wheel_joint_state = p.getJointState(self.bicycleId, self.fly_wheel_joint, self.client)
        fly_wheel_joint_ang = fly_wheel_joint_state[0] % (2 * math.pi)
        fly_wheel_joint_vel = fly_wheel_joint_state[1]

        observation = [roll_angle, roll_angular_vel, fly_wheel_joint_ang, fly_wheel_joint_vel]
        return observation

    def reset(self):
        p.resetBasePositionAndOrientation(self.bicycleId, self.initial_position, self.initial_orientation)
        p.resetJointState(self.bicycleId, self.handlebar_joint, targetValue=0, targetVelocity=0)
        return self.get_observation()
