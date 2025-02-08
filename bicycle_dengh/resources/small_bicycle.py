import pybullet as p
import random
import math
import os
import time
import platform
import pybullet_data
import keyboard


class SmallBicycle:
    def __init__(self, client, max_flywheel_vel):
        self.client = client

        system = platform.system()
        if system == "Windows":
            f_name = os.path.join(os.path.dirname(__file__), 'small_bicycle\\small_bicycle.xml')
        else:
            f_name = os.path.join(os.path.dirname(__file__), 'small_bicycle/small_bicycle.xml')

        startOrientation = p.getQuaternionFromEuler([0, 0, 1.57])
        self.bicycleId = p.loadURDF(fileName=f_name, basePosition=[0, 0, 1], baseOrientation=startOrientation, globalScaling=0.01)

        self.fly_wheel_joint = 7
        self.gyros_link = 8

        self.initial_joint_positions = None
        self.initial_joint_velocities = None
        self.initial_position, self.initial_orientation = p.getBasePositionAndOrientation(self.bicycleId)

        # 设置飞轮速度上限
        p.changeDynamics(self.bicycleId,
                         self.fly_wheel_joint,
                         maxJointVelocity=max_flywheel_vel,
                         physicsClientId=self.client)

    def apply_action(self, fly_wheel_speed):
        p.setJointMotorControl2(bodyUniqueId=self.bicycleId,
                                jointIndex=self.fly_wheel_joint,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=fly_wheel_speed,
                                force=1000,
                                physicsClientId=self.client)

    def get_observation(self):
        gyros_link_state = p.getLinkState(self.bicycleId, self.gyros_link, computeLinkVelocity=1,
                                          physicsClientId=self.client)
        gyros_link_orientation = gyros_link_state[1]
        link_ang = p.getEulerFromQuaternion(gyros_link_orientation)
        roll_angle = link_ang[0]
        # print(f"roll_angle: {roll_angle}")
        return [roll_angle]

    def reset(self):
        p.resetBasePositionAndOrientation(self.bicycleId, self.initial_position,
                                          self.initial_orientation, self.client)
        # p.resetJointState(self.bicycleId, self.handlebar_joint, 0, 0, self.client)
        p.resetJointState(self.bicycleId, self.fly_wheel_joint, 0, 0, self.client)
        # p.resetJointState(self.bicycleId, self.front_wheel_joint, 0, 0, self.client)
        # p.resetJointState(self.bicycleId, self.back_wheel_joint, 0, 0, self.client)
        return self.get_observation()


if __name__ == '__main__':
    client = p.connect(p.GUI)
    bicycle = SmallBicycle(client, 10)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.loadURDF("plane.urdf", physicsClientId=client)
    p.setGravity(0, 0, -10)
    p.setRealTimeSimulation(1)
    # p.setTimeStep(2e-4)
    # 获取机器人所有关节信息
    num_joints = p.getNumJoints(bicycle.bicycleId)
    joint_info = []

    for joint_index in range(num_joints):
        info = p.getJointInfo(bicycle.bicycleId, joint_index)
        joint_info.append({
            "index": info[0],
            "name": info[1].decode('utf-8'),
            "type": info[2],
            "lower_limit": info[8],
            "upper_limit": info[9]
        })
        print(f"Joint {info[0]}: {info[1].decode('utf-8')}")

    # 创建调试滑块参数用于控制关节
    slider_ids = []
    for joint in joint_info:
        if joint["type"] == p.JOINT_REVOLUTE:  # 只对旋转关节创建滑块
            slider_id = p.addUserDebugParameter(
                joint["name"],
                joint["lower_limit"] if joint["lower_limit"] > -1e10 else -3.14,
                joint["upper_limit"] if joint["upper_limit"] < 1e10 else 3.14,
                0.0
            )
            slider_ids.append((joint["index"], slider_id))

    while True:
        # 读取滑块参数并施加到机器人关节
        for joint_index, slider_id in slider_ids:
            target_position = p.readUserDebugParameter(slider_id)
            p.setJointMotorControl2(
                bodyUniqueId=bicycle.bicycleId,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_position
            )
        # bicycle.apply_action(0)
        bicycle.get_observation()

        link_state = p.getLinkState(bicycle.bicycleId, 8, computeForwardKinematics=True)
        link_world_position = link_state[0]
        link_world_orientation = link_state[1]

        # 绘制坐标轴
        p.addUserDebugLine(link_world_position,
                           [link_world_position[0] + 0.1, link_world_position[1], link_world_position[2]],
                           [1, 0, 0], 8)  # X轴，红色
        p.addUserDebugLine(link_world_position,
                           [link_world_position[0], link_world_position[1] + 0.1, link_world_position[2]],
                           [0, 1, 0], 8)  # Y轴，绿色
        p.addUserDebugLine(link_world_position,
                           [link_world_position[0], link_world_position[1], link_world_position[2] + 0.1],
                           [0, 0, 1], 8)  # Z轴，蓝色

        if keyboard.is_pressed('c'):
            p.removeAllUserDebugItems()
        p.stepSimulation()

