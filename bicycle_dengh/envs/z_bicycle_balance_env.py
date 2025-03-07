import gymnasium
import numpy as np
import pybullet as p
import pybullet_data
from bicycle_dengh.resources.z_bicycle import ZBicycle
import math
import csv
import time
from simple_pid import PID


class ZBicycleBalanceEnv(gymnasium.Env):
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, gui=False):
        self.terminated = False
        self.truncated = False
        self.gui = gui
        self.max_flywheel_vel = 120.
        self._max_episode_steps = 1100
        self._elapsed_steps = None
        self.flywheel_vel_array = []
        self.roll_angle_array = []
        self.roll_angle_vel_array = []
        self.roll_angle_pid = PID(1000, 0, 0, setpoint=0.0)
        self.current_roll_angle = 0.0

        self.action_space = gymnasium.spaces.box.Box(low=-1., high=1., shape=(1,), dtype=np.float32)

        # 飞轮
        self.actual_action_space = gymnasium.spaces.box.Box(
            low=np.array([-self.max_flywheel_vel]),
            high=np.array([self.max_flywheel_vel]),
            shape=(1,),
            dtype=np.float32
        )

        self.action_low, self.action_high = self.actual_action_space.low, self.actual_action_space.high

        # 翻滚角, 翻滚角角速度, 飞轮速度
        self.observation_space = gymnasium.spaces.box.Box(
            low=np.array([-math.pi, -15.0, -self.max_flywheel_vel]),
            high=np.array([math.pi, 15.0, self.max_flywheel_vel]),
            shape=(3,),
            dtype=np.float32
        )

        if self.gui:
            self.client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

        self.bicycle = ZBicycle(self.client, self.max_flywheel_vel)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        # 更改地面物体的动力学参数，包括摩擦系数
        p.changeDynamics(plane_id, -1, lateralFriction=3.5)  # -1表示所有部件
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.setGravity(0, 0, -10, physicsClientId=self.client)
        p.setTimeStep(1. / 10., self.client)

        # 定义方块的初始位置
        initial_position = [-0.3, -0.1, 1.5]  # [x, y, z] 坐标

        # 定义方块的初始姿态（四元数）
        initial_orientation = p.getQuaternionFromEuler([0, 0, 0])

        # 创建一个立方体形状
        boxHalfExtents = [0.1, 0.1, 0.1]  # 方块的半边长
        visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX,
                                            halfExtents=boxHalfExtents)
        collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                  halfExtents=boxHalfExtents)

        # 创建刚体（方块）
        # boxId = p.createMultiBody(baseMass=1,
        #                           baseCollisionShapeIndex=collisionShapeId,
        #                           baseVisualShapeIndex=visualShapeId,
        #                           basePosition=initial_position,
        #                           baseOrientation=initial_orientation,
        #                           physicsClientId=self.client)

    def step(self, action):
        rescaled_action = self._rescale_action(action)
        roll_angle_control = self.roll_angle_pid(self.current_roll_angle)
        input_action = [0, 0, rescaled_action[0]]
        # input_action = [0, 0, -roll_angle_control]

        self.bicycle.apply_action(input_action)
        p.stepSimulation(physicsClientId=self.client)

        obs = self.bicycle.get_observation()

        self.roll_angle_array.append(obs[0])
        self.roll_angle_vel_array.append(obs[1])
        self.flywheel_vel_array.append(obs[2])

        ret_obs = [obs[0], obs[1], obs[2]]
        self.current_roll_angle = obs[0]
        reward = self._reward_fun(ret_obs)

        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            # print(f"平衡精度={np.mean(np.abs(self.roll_angle_array))}")
            # print(f"平衡稳定性={np.std(self.roll_angle_vel_array)}")
            # print(f"平衡能耗={np.mean(np.abs(self.flywheel_vel_array))}")
            self.truncated = True

        if self._elapsed_steps == 1000:
            with open(
                    'D:\data\\1-L\\9-bicycle\\bicycle-rl\exp_data\平衡实验数据处理\倾斜角实验\\roll_angle_ppo.csv',
                    'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['roll_angle'])  # 写入 CSV 文件的头部 写入列名，第一行
                # 将每个浮点数转换为包含该浮点数的列表
                rows = [[angle] for angle in self.roll_angle_array]
                csv_writer.writerows(rows)  # writerows 可以一次写入多行
                print("roll_angle_ppo.csv 写入完成")
        elif 700 <= self._elapsed_steps <= 701:
            self.bicycle.apply_lateral_disturbance(30.0, [-0.3, -0.0, 1.5])
            print("施加扰动")

        return np.array(ret_obs, dtype=np.float32), reward, self.terminated, self.truncated, {}

    def reset(self, seed=None, options=None):
        self.terminated = False
        self.truncated = False
        self._elapsed_steps = 0

        self.flywheel_vel_array = []
        self.roll_angle_array = []
        self.roll_angle_vel_array = []

        obs = self.bicycle.reset()
        ret_obs = [obs[0], obs[1], obs[2]]

        self.current_roll_angle = obs[0]

        return np.array(ret_obs, dtype=np.float32), {}

    def _reward_fun(self, obs):
        self.terminated = False
        self.truncated = False

        roll_angle = obs[0]
        roll_angle_vel = obs[1]
        wheel_vel = obs[2]

        r1 = 0.0
        if math.fabs(roll_angle) >= 0.35:
            self.terminated = True
            r1 = -10.0
        else:
            r1 = 1.0 - (math.fabs(roll_angle) / 0.35) * 2.0
            r1 = max(-1.0, min(1.0, r1))

        r2 = -0.1 * math.fabs(roll_angle_vel)
        r3 = -0.001 * math.fabs(wheel_vel)
        # print(f"r1: {r1}, r2: {r2}, r3: {r3}")

        return r1 + r2

    def render(self):
        pass

    def _rescale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high] (no need for symmetric action space)
        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        return self.action_low + (0.5 * (scaled_action + 1.0) * (self.action_high - self.action_low))


if __name__ == '__main__':
    env = gymnasium.make('ZBicycleBalanceEnv-v0', gui=True)
    obs, _ = env.reset()
    for i in range(40000):
        action = np.array([0.0], np.float32)
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()
        time.sleep(1. / 24.)

    env.close()
