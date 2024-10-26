import pybullet as p
import pybullet_data

# 启动仿真环境
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# 设置仿真环境中的重力
p.setGravity(0, 0, -9.8)

# 加载平面，便于观察墙体效果
plane_id = p.loadURDF("plane.urdf")

# 围墙的参数
wall_length = 5    # 墙体的长度
wall_height = 2    # 墙体的高度
wall_thickness = 0.1  # 墙体的厚度

# 创建墙体的碰撞形状和视觉形状
collision_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[wall_length / 2, wall_thickness / 2, wall_height / 2])
visual_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[wall_length / 2, wall_thickness / 2, wall_height / 2], rgbaColor=[1, 0, 0, 1])

# 指定墙体的位置
wall_position = [0, 2, wall_height / 2]  # x, y, z 位置
wall_orientation = p.getQuaternionFromEuler([0, 0, 0])  # 旋转角度，单位是弧度

# 创建墙体
wall_id = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collision_shape, baseVisualShapeIndex=visual_shape, basePosition=wall_position, baseOrientation=wall_orientation)

# 运行一段时间以便观察
while True:
    p.stepSimulation()

# 断开连接
p.disconnect()
