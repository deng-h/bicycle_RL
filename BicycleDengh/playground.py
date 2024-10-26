import pybullet as p
import time
import pybullet_data
from BicycleDengh.bicycle_dengh.resources import bicycle
from BicycleDengh.bicycle_dengh.resources import balance_bicycle
from simple_pid import PID

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -98)

# 返回值是一个整数，而不是一个可修改的对象
# 这个整数是传递给其他函数的ID，用于查询机器人的状态并对其执行操作
planeId = p.loadURDF("plane.urdf")

bicycle_vel = p.addUserDebugParameter('bicycle_vel', 0.0, 20.0, 10.0)
bicycle_handlebar = p.addUserDebugParameter('bicycle_handlebar', -1.57, 1.57, 0)
bicycle_flywheel = p.addUserDebugParameter('bicycle_flywheel', -20, 20, 0)

roll_angle_pid = PID(2500, 100, 100, setpoint=0.0)  # 初始翻滚角为0.0°的情况

camera_distance = 3
camera_yaw = 0
camera_pitch = -45
camera_target_position = [0, 0, 0]
p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)

b = bicycle.Bicycle(client=physicsClient, max_flywheel_vel=200)
# b = balance_bicycle.BalanceBicycle(client=physicsClient)

number_of_joints = p.getNumJoints(b.bicycleId)
print(f"共有{number_of_joints}个关节")
for joint_number in range(number_of_joints):
    joint_info = p.getJointInfo(b.bicycleId, joint_number)
    joint_index = joint_info[0]
    joint_name = joint_info[1].decode('utf-8')
    link_name = joint_info[12].decode('utf-8')
    print(f"jointIndex={joint_index}, jointName={joint_name}, linkName={link_name}")

try:
    while True:
        bicycle_vel_param = p.readUserDebugParameter(bicycle_vel)
        bicycle_handlebar_param = p.readUserDebugParameter(bicycle_handlebar)
        bicycle_flywheel_param = p.readUserDebugParameter(bicycle_flywheel)
        # b.apply_action([0, 0, 1.0])
        obs = b.get_observation()
        # print(obs[0], obs[1], obs[2])
        roll_angle_control = roll_angle_pid(obs[0])
        # action = [-roll_angle_control]
        action = [0.0, 0.0, -roll_angle_control]
        # b.apply_action(action)
        p.stepSimulation()
        time.sleep(1. / 240.)
finally:
    p.disconnect()
