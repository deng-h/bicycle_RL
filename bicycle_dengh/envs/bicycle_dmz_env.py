import time
import gymnasium
import numpy as np
import pybullet as p
import pybullet_data
import random
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from bicycle_dengh.resources.bicycle_dmz import BicycleDmz
from bicycle_dengh.resources.goal import Goal
from playground.a_start.create_grid_map import create_grid_map2
from playground.a_start.create_obstacle import create_obstacle
from playground.a_start.visualize_path import visualize_path
from playground.a_start.get_goal_pos import get_goal_pos

import math
from utils import my_tools
from simple_pid import PID
import yaml
import platform


class BicycleDmzEnv(gymnasium.Env):
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, gui=False):
        system = platform.system()
        if system == "Windows":
            yaml_file_path = "D:\\data\\1-L\\9-bicycle\\bicycle-rl\\bicycle_dengh\envs\BicycleMazeLidarEnvConfig.yaml"
        else:
            yaml_file_path = "/root/bicycle-rl/bicycle_dengh/envs/BicycleMazeLidarEnvConfig.yaml"
        with open(yaml_file_path, "r", encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.goal = (1, 1)
        self.terminated = False
        self.truncated = False
        self.gui = gui
        self.max_flywheel_vel = 120.
        self.prev_goal_id = None
        self.prev_dist_to_goal = 0.
        self.prev_bicycle_pos = (0, 0)
        self.prev_center_angle = math.pi / 2
        self.prev_yaw_angle = math.pi / 2
        self._max_episode_steps = self.config["max_episode_steps"]
        self.goal_threshold = self.config["goal_threshold"]
        self.safe_distance = self.config["safe_distance"]
        self._elapsed_steps = None
        self.center_radius = 2.5
        self.episode_rwd = {"angle_penalty": 0.0, "navigation": 0.0, "obstacle": 0.0}
        self.action_space = gymnasium.spaces.box.Box(low=-1., high=1., shape=(1,), dtype=np.float32)

        # 1 改变action_space的范围
        # 2 研究PP的前视距离和self.center_radius
        # 3 prev_center_angle
        # 4 车把值滤波
        # 尝试逐步增加纯跟踪算法的前视距离 Ld (Lookahead Distance) 的值，观察 steering_angle 的变化
        # 找到一个既能保证跟踪性能，又能使 steering_angle 保持平稳的折衷值
        # 限制车把转速 changeDynamics
        # 以自行车为中心的扇形区域
        self.actual_action_space = gymnasium.spaces.box.Box(
            low=np.array([-1.4]),  # 1.4弧度 80.4度
            high=np.array([1.4]),
            shape=(1,),
            dtype=np.float32)

        self.action_low, self.action_high = self.actual_action_space.low, self.actual_action_space.high

        # 机器人与目标点距离, 机器人与目标点的角度, 翻滚角, 翻滚角角速度, 车把角度, 车把角速度, 后轮速度, 飞轮速度
        self.observation_space = gymnasium.spaces.Dict({
            "lidar": gymnasium.spaces.box.Box(low=0., high=50., shape=(120,), dtype=np.float32),
            "bicycle": gymnasium.spaces.box.Box(
                low=np.array([0.0, -math.pi, -math.pi, -15.0, -1.57, -30.0, 0.0, -self.max_flywheel_vel]),
                high=np.array([100.0, math.pi, math.pi, 15.0, 1.57, 30.0, 10.0, self.max_flywheel_vel]),
                shape=(8,),
                dtype=np.float32),
        })

        if self.gui:
            self.client = p.connect(p.GUI)
            self.camera_distance_param = p.addUserDebugParameter('camera_distance_param', 2, 60, 5)
            self.camera_yaw_param = p.addUserDebugParameter('camera_yaw_param', -180, 180, 0)
            self.camera_pitch_param = p.addUserDebugParameter('camera_pitch_param', -90, 90, -30)
            self.bicycle_vel_param = p.addUserDebugParameter('bicycle_vel_param', 0.0, 3.0, 1.0)
            self.handlebar_angle_param = p.addUserDebugParameter('handlebar_angle_param', -1.57, 1.57, 0)
            self.flywheel_param = p.addUserDebugParameter('flywheel_param', -40, 40, 0)
        else:
            self.client = p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

        # obstacle_ids = create_obstacle(self.client)
        self.bicycle = BicycleDmz(self.client, self.max_flywheel_vel, obstacle_ids=[])
        self.bicycle.init_pure_pursuit_controller(self.bicycle)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        # 更改地面物体的动力学参数，包括摩擦系数，-1表示所有部件
        p.changeDynamics(plane_id, -1, lateralFriction=1.5)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.setGravity(0, 0, -10, physicsClientId=self.client)
        # p.setRealTimeSimulation(0)
        # p.setTimeStep(1. / 150., self.client)

    def step(self, action):
        rescaled_action = self._rescale_action(action)
        # fly_wheel_action = rescaled_action[0]
        center_angle = rescaled_action[0]
        # print(f"center_angle: {center_angle:.3f}")
        pure_pursuit_point = self._calculate_new_position(self.prev_bicycle_pos[0],
                                                          self.prev_bicycle_pos[1],
                                                          self.prev_yaw_angle,
                                                          self.center_radius, center_angle)
        # print(f"自行车坐标: {self.prev_bicycle_pos}, center_angle: {center_angle:.3f}, pure_pursuit_point: {pure_pursuit_point}")
        self.bicycle.apply_action3(fly_wheel_action=0.0, points=[pure_pursuit_point])
        p.stepSimulation(physicsClientId=self.client)
        obs = self.bicycle.get_observation()
        distance_to_goal = np.linalg.norm(np.array([obs[0], obs[1]]) - np.array(self.goal))
        angle_to_target = my_tools.calculate_angle_to_target(obs[0], obs[1], obs[2], self.goal[0], self.goal[1])
        bicycle_obs = np.array([distance_to_goal, angle_to_target, obs[3], obs[4], obs[5], obs[6], obs[7], obs[8]],
                               dtype=np.float32)

        if self.gui:
            bike_pos, _ = p.getBasePositionAndOrientation(self.bicycle.bicycleId, physicsClientId=self.client)
            camera_distance = p.readUserDebugParameter(self.camera_distance_param)
            camera_yaw = p.readUserDebugParameter(self.camera_yaw_param)
            camera_pitch = p.readUserDebugParameter(self.camera_pitch_param)
            # p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-85, cameraTargetPosition=[15, 10, 10])
            p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, bike_pos)

        delta_angle = center_angle - self.prev_center_angle
        reward = self._reward_fun(bicycle_obs, lidar_data=obs[9], delta_angle=delta_angle, is_collision=obs[10])

        self.prev_center_angle = center_angle
        self.prev_dist_to_goal = distance_to_goal
        self.prev_bicycle_pos = (obs[0], obs[1])
        self.prev_yaw_angle = obs[2]

        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            self.truncated = True

        return {"lidar": obs[9], "bicycle": bicycle_obs}, reward, self.terminated, self.truncated, {}

    def reset(self, seed=None, options=None):
        self.terminated = False
        self.truncated = False
        self._elapsed_steps = 0
        # print(self.episode_rwd)
        self.episode_rwd = {"angle_penalty": 0.0, "navigation": 0.0, "obstacle": 0.0}

        x = random.randint(-3, 3)
        y = random.randint(15, 25)
        self.goal = (x, y)
        goal = Goal(self.client, self.goal)
        if self.prev_goal_id is not None:
            p.removeBody(self.prev_goal_id, self.client)
        self.prev_goal_id = goal.id

        obs = self.bicycle.reset()
        distance_to_goal = np.linalg.norm(np.array([obs[0], obs[1]]) - np.array(self.goal))
        angle_to_target = my_tools.calculate_angle_to_target(obs[0], obs[1], obs[2], self.goal[0], self.goal[1])

        bicycle_obs = np.array([distance_to_goal, angle_to_target, obs[3], obs[4], obs[5], obs[6], obs[7], obs[8]],
                               dtype=np.float32)

        self.prev_dist_to_goal = distance_to_goal
        self.prev_bicycle_pos = (obs[0], obs[1])
        self.prev_center_angle = math.pi / 2
        self.prev_yaw_angle = math.pi / 2

        return {"lidar": obs[9], "bicycle": bicycle_obs}, {}

    def _reward_fun(self, obs, lidar_data, delta_angle, is_collision=False):
        self.terminated = False
        self.truncated = False
        # action 车把角度，前后轮速度, 飞轮速度, 相对路径点的距离x, 相对路径点的距离y
        # obs 机器人与目标点距离, 机器人与目标点的角度, 翻滚角, 翻滚角角速度, 车把角度, 车把角速度, 后轮速度, 飞轮速度
        distance_to_goal = obs[0]
        roll_angle = obs[2]

        # ========== 平衡奖励 ==========
        balance_rwd = 0.0
        if math.fabs(roll_angle) >= 0.45:
            self.terminated = True
            balance_rwd = -120.0
        # else:
        #     balance_rwd = 1.0 - (math.fabs(roll_angle) / 0.35) * 2.0
        #     balance_rwd = max(-1.0, min(1.0, balance_rwd))
        # ========== 平衡奖励 ==========

        # ========== 导航奖励 ==========
        diff_dist = (self.prev_dist_to_goal - distance_to_goal) * 2.5
        distance_rwd = diff_dist if diff_dist > 0 else 1.2 * diff_dist

        goal_rwd = 0.0
        if distance_to_goal <= self.goal_threshold:
            print("=====到达目标点=====")
            self.terminated = True
            goal_rwd = 1000.0

        navigation_rwd = distance_rwd + goal_rwd
        # ========== 导航奖励 ==========

        # ========== 避障奖励 ==========
        # obstacle_penalty = 0.0
        # min_obstacle_dist = np.min(lidar_data)
        # if min_obstacle_dist <= self.safe_distance:
        #     obstacle_penalty = -10.0
        # avoid_obstacle_rwd = obstacle_penalty
        # ========== 避障奖励 ==========

        angle_penalty = 0.0
        if math.fabs(delta_angle) >= 0.17:
            angle_penalty = -0.5
        # angle_penalty = -0.25 * math.fabs(delta_angle)
        # print(f"distance_rwd: {distance_rwd:.5f}, "
        #       f"goal_rwd: {goal_rwd:.5f}, "
        #       f"angle_penalty: {angle_penalty:.5f}")

        # total_reward = balance_rwd + navigation_rwd + avoid_obstacle_rwd
        # self.episode_rwd["angle_penalty"] += angle_penalty
        # self.episode_rwd["navigation"] += navigation_rwd

        total_reward = balance_rwd + navigation_rwd + angle_penalty

        return total_reward

    def render(self):
        pass

    def close(self):
        p.disconnect(self.client)

    def _rescale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high] (no need for symmetric action space)
        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        return self.action_low + (0.5 * (scaled_action + 1.0) * (self.action_high - self.action_low))

    def _calculate_new_position(self, x, y, yaw, distance, angle):
        """
        根据当前位置、距离和角度计算新的位置。

        参数:
            x (float): 当前机器人的 x 坐标。
            y (float): 当前机器人的 y 坐标。
            distance (float): 距离。
            angle (float): 随机生成的角度（弧度制，范围 -π 到 π）。

        返回:
            tuple: 新的机器人位置 (new_x, new_y)。
        """
        # 目标点的角度是自行车yaw角度加上约束后的外部角度
        target_angle = yaw + angle

        # 使用三角函数计算新位置
        delta_x = distance * math.cos(target_angle)
        delta_y = distance * math.sin(target_angle)

        # 计算新坐标
        new_x = x + delta_x
        new_y = y + delta_y

        return new_x, new_y


if __name__ == '__main__':
    env = gymnasium.make('BicycleDmzEnv-v0', gui=True)
    obs, _ = env.reset()
    # check_observation_space(obs, env.observation_space)
    for i in range(10000):
        action = np.array([0], np.float32)
        obs, _, terminated, truncated, infos = env.step(action)

        # if terminated or truncated:
        #     obs, _ = env.reset()
        time.sleep(10000)

    env.close()
