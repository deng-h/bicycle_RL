import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from bicycle_dengh.resources.bicycle import Bicycle
from bicycle_dengh.resources.goal import Goal
from bicycle_dengh.resources.wall import Wall
import math
import random


class BicycleDenghEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, gui=False):
        self.bicycle = None
        self.goal = (0, 0)
        self.terminated = False
        self.truncated = False
        self.prev_dist_to_goal = 0.0
        self.prev_action = [0.0, 0.0, 0.0]
        self.gui = gui
        # 限制倾斜角度
        self.roll_angle_epsilon = 0.3
        self.max_flywheel_vel = 200.0

        # action_space[车把角度，前后轮速度，飞轮转速]
        self.action_space = gym.spaces.box.Box(
            low=np.array([-1.57, -5.0, -self.max_flywheel_vel]),
            high=np.array([1.57, 5.0, self.max_flywheel_vel]),
            shape=(3,),
            dtype=np.float32)

        # [pos[0], pos[1],roll_angle, roll_vel,handlebar_joint_ang, handlebar_joint_vel,
        #  back_wheel_joint_ang, back_wheel_joint_vel,fly_wheel_joint_ang, fly_wheel_joint_vel]
        self.observation_space = gym.spaces.box.Box(
            low=np.array([0.0, -3.14, -10.0, -1.57, -10.0, -10.0]),
            high=np.array([35.0, 3.14, 10.0, 1.57, 10.0, 10.0]),
            shape=(6,),
            dtype=np.float32)

        if gui:
            self.client = p.connect(p.GUI)
            self.camera_distance_param = p.addUserDebugParameter('camera_distance_param', 2, 60, 2)
            self.camera_yaw_param = p.addUserDebugParameter('camera_yaw_param', -180, 180, 0)
            self.camera_pitch_param = p.addUserDebugParameter('camera_pitch_param', -90, 90, -25)
        else:
            self.client = p.connect(p.DIRECT)

        p.setTimeStep(1. / 24., self.client)
        self.bicycle_vel_param = p.addUserDebugParameter('bicycle_vel_param', 0.0, 3.0, 1.0)
        self.handlebar_angle_param = p.addUserDebugParameter('handlebar_angle_param', -1.57, 1.57, 0)
        self.flywheel_param = p.addUserDebugParameter('flywheel_param', -40, 40, 0)

    def step(self, action):
        self.bicycle.apply_action(action)
        p.stepSimulation(physicsClientId=self.client)
        obs = self.bicycle.get_observation()

        if self.gui:
            bike_pos, _ = p.getBasePositionAndOrientation(self.bicycle.bicycleId, physicsClientId=self.client)
            camera_distance = p.readUserDebugParameter(self.camera_distance_param)
            camera_yaw = p.readUserDebugParameter(self.camera_yaw_param)
            camera_pitch = p.readUserDebugParameter(self.camera_pitch_param)
            p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, bike_pos)

        dis_x = self.goal[0] - obs[0]
        dis_y = self.goal[1] - obs[1]
        dis_to_goal = math.sqrt(dis_x ** 2 + dis_y ** 2)

        # 计算奖励值
        reward = self._reward_fun(obs, dis_to_goal)

        obs = np.array([dis_to_goal, obs[2], obs[3], obs[4], obs[5], obs[6]], dtype=np.float32)

        return obs, reward, self.terminated, self.truncated, {}

    def reset(self, seed=None, options=None):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -10, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self.client)

        self.terminated = False
        self.truncated = False

        self.bicycle = Bicycle(client=self.client)
        # 设置飞轮速度上限
        p.changeDynamics(self.bicycle.bicycleId,
                         self.bicycle.fly_wheel_joint,
                         maxJointVelocity=self.max_flywheel_vel,
                         physicsClientId=self.client)

        # 设置目标点
        x = (random.uniform(10, 20) if random.choice([True, False]) else random.uniform(-20, -10))
        y = (random.uniform(10, 20) if random.choice([True, False]) else random.uniform(-20, -10))
        self.goal = (x, y)
        Goal(self.client, self.goal)

        # Wall(self.client, [0, 0.7, 0])
        # Wall(self.client, [0, -0.7, 0])

        obs = self.bicycle.get_observation()
        dis_x = self.goal[0] - obs[0]
        dis_y = self.goal[1] - obs[1]
        dis_to_goal = math.sqrt(dis_x ** 2 + dis_y ** 2)
        self.prev_dist_to_goal = math.sqrt(((obs[0] - self.goal[0]) ** 2 + (obs[1] - self.goal[1]) ** 2))
        return np.array([dis_to_goal, obs[2], obs[3], obs[4], obs[5], obs[6]], dtype=np.float32), {}

    def _reward_fun(self, obs, dist_to_goal):
        self.terminated = False
        self.truncated = False

        roll_angle = obs[2]
        if math.fabs(roll_angle - math.pi / 2) > self.roll_angle_epsilon:
            self.terminated = True
            balance_reward = -1.0
        else:
            balance_reward = 1.0

        # 越界惩罚
        if obs[0] >= 35.0 or obs[0] <= -35.0 or obs[1] >= 35.0 or obs[1] <= -35.0:
            self.terminated = True
            bound_reward = -10.0
        else:
            bound_reward = 0.0

        # 达到目标点奖励
        # D是靠近目的地阈值，当自行车到目标点的距离小于D时，即不受到惩罚，当离目标点的距离大于D时，随着距离越大，惩罚值越大
        D = 5.0
        reach_goal_reward = -max(dist_to_goal - D, 0.0)
        # 当自行车靠近目标附近的一个小区域时，增加奖励值使自行车加速接近目标点
        k1 = 0.5
        k2 = -1.0
        # additional_reward = k1 * reach_goal_reward + k2 * math.log(reach_goal_reward + 1e-6, math.e)

        total_reward = balance_reward + bound_reward

        return total_reward

    def render(self):
        pass

    def close(self):
        p.disconnect(self.client)
