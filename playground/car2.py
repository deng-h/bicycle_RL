import keyboard
import numpy as np
import pybullet as p
import time
import pybullet_data
import matplotlib.pyplot as plt

from utils import my_tools

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.resetSimulation()
p.setGravity(0, 0, -10)
useRealTimeSim = 1

# for video recording (works best on Mac and Linux, not well on Windows)
# p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, "racecar.mp4")
p.setRealTimeSimulation(useRealTimeSim)  # either this
p.loadURDF("plane.urdf")
car = p.loadURDF("racecar/racecar_differential.urdf", [0, 10, 0.2])
obstacle_ids = my_tools.build_maze(physicsClient)

for i in range(p.getNumJoints(car)):
    print(p.getJointInfo(car, i))
for wheel in range(p.getNumJoints(car)):
    p.setJointMotorControl2(car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
    p.getJointInfo(car, wheel)

wheels = [8, 15]
print("----------------")

# p.setJointMotorControl2(car,10,p.VELOCITY_CONTROL,targetVelocity=1,force=10)
c = p.createConstraint(car, 9, car, 11, jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0],
                       childFramePosition=[0, 0, 0])
p.changeConstraint(c, gearRatio=1, maxForce=10000)

c = p.createConstraint(car, 10, car, 13, jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0],
                       childFramePosition=[0, 0, 0])
p.changeConstraint(c, gearRatio=-1, maxForce=10000)

c = p.createConstraint(car, 9, car, 13, jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0],
                       childFramePosition=[0, 0, 0])
p.changeConstraint(c, gearRatio=-1, maxForce=10000)

c = p.createConstraint(car, 16, car, 18, jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0],
                       childFramePosition=[0, 0, 0])
p.changeConstraint(c, gearRatio=1, maxForce=10000)

c = p.createConstraint(car, 16, car, 19, jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0],
                       childFramePosition=[0, 0, 0])
p.changeConstraint(c, gearRatio=-1, maxForce=10000)

c = p.createConstraint(car, 17, car, 19, jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0],
                       childFramePosition=[0, 0, 0])
p.changeConstraint(c, gearRatio=-1, maxForce=10000)

c = p.createConstraint(car, 1, car, 18, jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0],
                       childFramePosition=[0, 0, 0])
p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)
c = p.createConstraint(car, 3, car, 19, jointType=p.JOINT_GEAR, jointAxis=[0, 1, 0], parentFramePosition=[0, 0, 0],
                       childFramePosition=[0, 0, 0])
p.changeConstraint(c, gearRatio=-1, gearAuxLink=15, maxForce=10000)

steering = [0, 2]

targetVelocitySlider = p.addUserDebugParameter("wheelVelocity", -50, 50, 0)
maxForceSlider = p.addUserDebugParameter("maxForce", 0, 50, 20)
steeringSlider = p.addUserDebugParameter("steering", -1, 1, 0)

targetVelocity = 0.0
steeringAngle = 0.0

# 激光雷达参数
lidar_range = 75.0  # 激光雷达的最大探测距离
num_rays = 720  # 扫描线数量
lidar_origin_offset = [0, 0, 1.0]  # 激光雷达相对于小车的位置偏移量
angle_step = np.deg2rad(360 / num_rays)  # 每条射线的角度增量

# 设置激光雷达的方向
ray_angles = np.linspace(-np.pi, np.pi, num_rays)
ray_directions = [[np.cos(angle), np.sin(angle), 0] for angle in ray_angles]
ray_to_positions = [[lidar_origin_offset[0] + lidar_range * dir[0],
                     lidar_origin_offset[1] + lidar_range * dir[1],
                     lidar_origin_offset[2]] for dir in ray_directions]

while True:
    maxForce = p.readUserDebugParameter(maxForceSlider)
    # targetVelocity = p.readUserDebugParameter(targetVelocitySlider)
    # steeringAngle = p.readUserDebugParameter(steeringSlider)
    car_pos = p.getBasePositionAndOrientation(car)[0]
    lidar_origin = [car_pos[0] + lidar_origin_offset[0], car_pos[1] + lidar_origin_offset[1],
                    car_pos[2] + lidar_origin_offset[2]]

    results = p.rayTestBatch([lidar_origin] * num_rays, ray_to_positions)
    distances = []
    for i, result in enumerate(results):
        if result[2] != -1:
            # 获取到碰撞点，计算激光雷达到碰撞点的距离
            hit_position = result[3]
            distance = np.linalg.norm(np.array(hit_position) - np.array(lidar_origin))
            # 从 lidar_origin 到碰撞点画线
            # p.addUserDebugLine(lidar_origin, hit_position, lineColorRGB=[1, 0, 0], lineWidth=1.0)
        else:
            # 未检测到碰撞，设为最大探测距离
            distance = lidar_range
            # 没有碰撞，绘制最大距离的射线
            # end_position = [lidar_origin[0] + lidar_range * ray_directions[i][0],
            #                 lidar_origin[1] + lidar_range * ray_directions[i][1],
            #                 lidar_origin[2]]
            # p.addUserDebugLine(lidar_origin, end_position, lineColorRGB=[0, 1, 0], lineWidth=1.0)
        # if i % 5 == 0:
        #     p.removeAllUserDebugItems()
        distances.append(distance)

    if keyboard.is_pressed('q'):
        # 生成激光雷达图
        x_vals = [distances[i] * np.cos(ray_angles[i]) for i in range(num_rays)]
        y_vals = [distances[i] * np.sin(ray_angles[i]) for i in range(num_rays)]
        # 绘制激光雷达图
        plt.figure(figsize=(8, 8))
        plt.plot(x_vals, y_vals, 'o', markersize=2)
        plt.xlim(-lidar_range, lidar_range)
        plt.ylim(-lidar_range, lidar_range)
        plt.xlabel('X (meters)')
        plt.ylabel('Y (meters)')
        plt.title('Simulated LiDAR Scan using PyBullet')
        plt.grid()
        plt.show()
    elif keyboard.is_pressed('up'):
        targetVelocity = 10.0
    elif keyboard.is_pressed('down'):
        targetVelocity = -10.0
    elif keyboard.is_pressed('left'):
        steeringAngle = -0.5
    elif keyboard.is_pressed('right'):
        steeringAngle = 0.5

    for wheel in wheels:
        p.setJointMotorControl2(car, wheel, p.VELOCITY_CONTROL, targetVelocity=targetVelocity, force=maxForce)

    for steer in steering:
        p.setJointMotorControl2(car, steer, p.POSITION_CONTROL, targetPosition=-steeringAngle)

    if useRealTimeSim == 0:
        p.stepSimulation()
    time.sleep(0.01)
