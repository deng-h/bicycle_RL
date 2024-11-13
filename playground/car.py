import pybullet as p
import pybullet_data
import numpy as np
import time

p.connect(p.GUI)

p.setAdditionalSearchPath(pybullet_data.getDataPath())  # 设置搜索路径
racecar_id = p.loadURDF("racecar/racecar_differential.urdf", basePosition=[0, 0, 0.1])
p.loadURDF("plane.urdf")

p.setGravity(0, 0, -9.8)

num_joints = p.getNumJoints(racecar_id)
print("Number of joints: ", num_joints)
for i in range(num_joints):
    # 获取自行车每个部分的ID
    joint_info = p.getJointInfo(racecar_id, i)
    print("jointIndex: ", joint_info[0], "jointName: ", joint_info[1])


# 设置车轮的摩擦力
# wheel_ids = [2, 3, 4, 5]  # racecar_differential中车轮的ID
# for wheel_id in wheel_ids:
#     p.setJointMotorControl2(racecar_id, wheel_id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

left_front_wheel_joint = 1
right_front_wheel_joint = 3
left_rear_wheel_joint = 12
right_rear_wheel_joint = 14

# 激光雷达参数
lidar_range = 5.0  # 激光雷达的最大探测距离
num_rays = 360  # 扫描线数量
lidar_origin_offset = [0, 0, 0.3]  # 激光雷达相对于小车的位置偏移量
angle_step = np.deg2rad(360 / num_rays)  # 每条射线的角度增量


# 获取小车的位置
def get_car_position():
    return p.getBasePositionAndOrientation(racecar_id)[0]


# 设置激光雷达的方向
ray_angles = np.linspace(-np.pi, np.pi, num_rays)
ray_directions = [[np.cos(angle), np.sin(angle), 0] for angle in ray_angles]
ray_to_positions = [[lidar_origin_offset[0] + lidar_range * dir[0],
                     lidar_origin_offset[1] + lidar_range * dir[1],
                     lidar_origin_offset[2]] for dir in ray_directions]


# 画雷达扫描的函数
def draw_lidar():
    car_pos = get_car_position()  # 获取小车当前位置
    lidar_origin = [car_pos[0] + lidar_origin_offset[0], car_pos[1] + lidar_origin_offset[1],
                    car_pos[2] + lidar_origin_offset[2]]

    results = p.rayTestBatch([lidar_origin] * num_rays, ray_to_positions)

    for i, result in enumerate(results):
        if result[0] != -1:  # 碰撞检测成功
            hit_position = result[3]
            p.addUserDebugLine(lidar_origin, hit_position, lineColorRGB=[1, 0, 0], lineWidth=1.0)
        else:  # 没有碰撞，显示最大距离
            end_position = [lidar_origin[0] + lidar_range * ray_directions[i][0],
                            lidar_origin[1] + lidar_range * ray_directions[i][1],
                            lidar_origin[2]]
            p.addUserDebugLine(lidar_origin, end_position, lineColorRGB=[0, 1, 0], lineWidth=1.0)


# 控制小车的函数
def control_car():
    keyboard_events = p.getKeyboardEvents()

    # 初始化车轮速度
    speed = 0
    turn = 0

    # 解析键盘输入
    if ord('q') in keyboard_events and keyboard_events[ord('q')] & p.KEY_WAS_TRIGGERED:
        speed = 5  # 向前
        print("speed=5")
    if ord('a') in keyboard_events and keyboard_events[ord('a')] & p.KEY_WAS_TRIGGERED:
        speed = -5  # 向后
        print("speed=-5")
    if ord('z') in keyboard_events and keyboard_events[ord('z')] & p.KEY_WAS_TRIGGERED:
        turn = 1  # 左转
        print("turn=1")
    if ord('x') in keyboard_events and keyboard_events[ord('x')] & p.KEY_WAS_TRIGGERED:
        turn = -1  # 右转
        print("turn=-1")

    # 控制车轮转向
    steering_angle = 0.5 * turn  # 左右转向角度
    p.setJointMotorControl2(racecar_id, 0, p.POSITION_CONTROL, targetPosition=steering_angle, force=20)  # 左前轮
    p.setJointMotorControl2(racecar_id, 1, p.POSITION_CONTROL, targetPosition=steering_angle, force=20)  # 右前轮


# 主循环
while True:
    # draw_lidar()  # 绘制雷达扫描
    # control_car()  # 控制小车运动

    p.setJointMotorControl2(racecar_id, 8, p.VELOCITY_CONTROL, targetVelocity=50, force=50)
    p.setJointMotorControl2(racecar_id, 15, p.VELOCITY_CONTROL, targetVelocity=50, force=50)

    p.stepSimulation()
    time.sleep(1. / 24.)  # 控制更新频率

