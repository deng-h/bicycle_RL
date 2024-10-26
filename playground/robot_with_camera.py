import pybullet
import pybullet_data

if __name__ == '__main__':
    # open the server
    physicsClient = pybullet.connect(pybullet.GUI)

    pybullet.setAdditionalSearchPath(pybullet_data.getDataPath())
    pybullet.setGravity(0, 0, -9.8)

    planeID = pybullet.loadURDF('plane.urdf')
    objectsID1 = pybullet.loadURDF('soccerball.urdf', [3, 0, 0])
    objectsID2 = pybullet.loadURDF('soccerball.urdf', [5, 0, 0])
    robotID = pybullet.loadURDF('r2d2.urdf', [0, 0, 1], [0, 0, 0, 1])

    camera_offset = [0.5, 0, 0.5]  # 相机相对于机器人的偏移 [前后, 左右, 上下]
    camera_up_vector = [0, 0, 1]  # 相机的上方向（z轴朝上）
    pybullet.setTimeStep(1. / 24., physicsClient)
    while True:
        position, orientation = pybullet.getBasePositionAndOrientation(robotID)
        # 更新相机的位置和方向
        camera_position = [
            position[0] + camera_offset[0],
            position[1] + camera_offset[1],
            position[2] + camera_offset[2]
        ]
        target_position = [position[0] + 3.0, position[1], position[2]]

        # 获取并渲染相机画面
        # DIRECT mode does allow rendering of images using the built-in software renderer
        # through the 'getCameraImage' API.
        # 也就是说开DIRECT模式也能获取图像
        # getCameraImage API 将返回一幅 RGB 图像、一个深度缓冲区和一个分割掩码缓冲区，其中每个像素都有可见物体的唯一 ID
        width, height, rgb_img, depth_img, seg_img = pybullet.getCameraImage(
            width=640,
            height=480,
            viewMatrix=pybullet.computeViewMatrix(
                cameraEyePosition=camera_position,  # 相机的实际位置，例如 [x, y, z] 坐标
                cameraTargetPosition=target_position,  # 相机所看的目标点位置，例如设置在相机前方的一点，通常与相机的前进方向一致
                cameraUpVector=camera_up_vector),  # 决定相机的“上”方向，例如 [0, 0, 1] 表示 z 轴为上。若要倾斜相机可以更改该向量
            # projectionMatrix定义了如何将三维场景投影到二维图像上，包括视野、长宽比和远近裁剪平面。可以理解为“拍摄效果的配置”
            projectionMatrix=pybullet.computeProjectionMatrixFOV(fov=60.0,  # 视野角度，角度越大视野越宽，但失真可能越明显
                                                                 aspect=1.0,  # 图像的宽高比，例如 640/480 或 1.0，确保图像不被拉伸或压缩
                                                                 # nearVal 和 farVal 决定了渲染图像的范围
                                                                 # 远近裁剪平面通常分别设置为 0.1 和 100，确保在视图中显示足够的景物而不出现异常裁剪
                                                                 nearVal=0.1,
                                                                 farVal=100.0)
        )

        # print(depth_img.shape)
        # print("================")
        pybullet.stepSimulation(physicsClient)

    # close server
    pybullet.disconnect()
