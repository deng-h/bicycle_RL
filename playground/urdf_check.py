import pybullet as p
import pybullet_data

def print_joint_indices(urdf_file):
    """
    加载URDF文件，并打印机器人的关节索引和相关信息。

    参数:
    urdf_file (str): URDF文件的路径。
    """
    physicsClient = p.connect(p.DIRECT) # 或 p.GUI
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0,0,-10) # 设置重力 (可选)

    robotId = p.loadURDF(urdf_file)

    numJoints = p.getNumJoints(robotId)
    print(f"机器人 '{urdf_file}' 共有 {numJoints} 个关节")

    for jointIndex in range(numJoints):
        jointInfo = p.getJointInfo(robotId, jointIndex)
        print(f"\n关节索引: {jointInfo[0]}")
        print(f"  关节名称: {jointInfo[1].decode('utf-8')}")
        # print(f"  关节类型: {jointInfo[2]}")
        # print(f"  关节轴向 (如果适用): {jointInfo[13]}")
        # print(f"  关节下限: {jointInfo[8]}")
        # print(f"  关节上限: {jointInfo[9]}")

    p.disconnect()

if __name__ == '__main__':
    urdf_file_path = "r2d2.urdf" #  您可以替换为您自己的URDF文件
    print_joint_indices(urdf_file_path)


    # 您也可以尝试加载其他的URDF文件，例如：
    # urdf_file_path = "laikago/laikago_gazebo.urdf" #  波士顿动力狗 (需要安装 laikago_gazebo 依赖)
    # print_joint_indices(urdf_file_path)

    #  如果您有自定义的URDF文件，请确保文件路径正确
    urdf_file_path = "D:\data\\1-L\9-bicycle\\bicycle-rl\\bicycle_dengh\\resources\\bicycle_urdf\\bike.xml"
    print_joint_indices(urdf_file_path)
