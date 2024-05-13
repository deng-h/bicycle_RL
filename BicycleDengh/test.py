import pybullet as p
import time
import pybullet_data
from BicycleDengh.bicycle_dengh.resources import bicycle

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -98)

# 返回值是一个整数，而不是一个可修改的对象
# 这个整数是传递给其他函数的ID，用于查询机器人的状态并对其执行操作
planeId = p.loadURDF("plane.urdf")

bicycle_vel = p.addUserDebugParameter('bicycle_vel', 0.0, 20.0, 10.0)
bicycle_handlebar = p.addUserDebugParameter('bicycle_handlebar', -1.57, 1.57, 0)
bicycle_flywheel = p.addUserDebugParameter('bicycle_flywheel', -20, 20, 0)

camera_distance = 3
camera_yaw = 0
camera_pitch = -45
camera_target_position = [0, 0, 0]
p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, camera_target_position)

bicycle = bicycle.Bicycle(client=physicsClient)

number_of_joints = p.getNumJoints(bicycle.bicycleId)
print(f"共有{number_of_joints}个关节")
for joint_number in range(number_of_joints):
    joint_info = p.getJointInfo(bicycle.bicycleId, joint_number)
    joint_index = joint_info[0]
    joint_name = joint_info[1]
    link_name = joint_info[12]
    print(f"jointIndex={joint_index}, jointName={joint_name.decode('utf-8')}, linkName={link_name.decode('utf-8')}")

i = 0
try:
    while True:
        # bicycle_vel = p.readUserDebugParameter(bicycle_vel)
        # bicycle_handlebar = p.readUserDebugParameter(bicycle_handlebar)
        # bicycle_flywheel = p.readUserDebugParameter(bicycle_flywheel)
        i += 1
        bicycle.apply_action([0, 0, 0])
        p.stepSimulation()
        time.sleep(1. / 240.)
finally:
    p.disconnect()
