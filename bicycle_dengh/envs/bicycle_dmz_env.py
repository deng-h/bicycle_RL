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
from playground.a_start.create_obstacle import generate_target_position


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
        self.center_radius = 2.0
        self.episode_rwd = {"1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0}
        self.pursuit_debug_point_id = None
        self.index = 0
        # 1 改变action_space的范围
        # 2 研究PP的前视距离和self.center_radius
        # 3 prev_center_angle
        # 4 车把值滤波
        # 尝试逐步增加纯跟踪算法的前视距离 Ld (Lookahead Distance) 的值，观察 steering_angle 的变化
        # 找到一个既能保证跟踪性能，又能使 steering_angle 保持平稳的折衷值
        # 限制车把转速 changeDynamics

        # 上层网络的引导角度（弧度）
        self.actual_action_space = gymnasium.spaces.box.Box(
            low=np.array([-1.4]),
            high=np.array([1.4]),
            shape=(1,),
            dtype=np.float32)

        self.action_low, self.action_high = self.actual_action_space.low, self.actual_action_space.high
        self.action_space = gymnasium.spaces.box.Box(low=-1., high=1., shape=(1,), dtype=np.float32)

        # 第一部分：60 个 [-12.0] ~ [12.0]
        num_groups = 60
        group1_low = np.array([-12.0])
        group1_high = np.array([12.0])
        # 使用 numpy.tile 快速创建重复的 low 和 high 数组
        part1_low = np.tile(group1_low, num_groups)
        part1_high = np.tile(group1_high, num_groups)
        # 第二部分：离目标点距离、离目标点角度、车身偏航角、车把角度
        part2_low = np.array([-50.0, -math.pi, -math.pi, -1.57])
        part2_high = np.array([50.0, math.pi, math.pi, 1.57])
        # 合并两部分 low 和 high 数组
        low = np.concatenate([part1_low, part2_low])
        high = np.concatenate([part1_high, part2_high])

        # 60个距离数据, 机器人与目标点距离, 机器人与目标点的角度, 车把角度, 车把角速度
        self.observation_space = gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)

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
        self.bicycle.init_pure_pursuit_controller(self.bicycle,
                                                  lookahead_distance=self.center_radius,
                                                  wheelbase=1.2)
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
        center_angle = rescaled_action[0]
        self.pursuit_point = self._calculate_new_position(self.prev_bicycle_pos[0],
                                                          self.prev_bicycle_pos[1],
                                                          self.prev_yaw_angle,
                                                          self.center_radius,
                                                          center_angle)
        self.bicycle.apply_action4(pursuit_pt=self.pursuit_point)
        p.stepSimulation(physicsClientId=self.client)

        obs = self.bicycle.get_observation()

        distance_to_goal = np.linalg.norm(np.array([obs[0], obs[1]]) - np.array(self.goal))
        angle_to_goal = my_tools.calculate_goal_angle_gemini(obs[0], obs[1], obs[2], self.goal[0], self.goal[1])
        processed_lidar_data = self._process_lidar_data(lidar_data=obs[9])
        bicycle_obs = np.array([distance_to_goal, angle_to_goal, obs[2], obs[5]], dtype=np.float32)
        ret_obs = np.concatenate((processed_lidar_data, bicycle_obs))

        if self.gui:
            bike_pos, _ = p.getBasePositionAndOrientation(self.bicycle.bicycleId, physicsClientId=self.client)
            camera_distance = p.readUserDebugParameter(self.camera_distance_param)
            camera_yaw = p.readUserDebugParameter(self.camera_yaw_param)
            camera_pitch = p.readUserDebugParameter(self.camera_pitch_param)
            # p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-85, cameraTargetPosition=[15, 10, 10])
            p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, bike_pos)
            # self.index += 1
            # if self.index % 30 == 0:
            #     # print(f"自行车坐标: {self.prev_bicycle_pos}, center_angle: {center_angle:.3f}, pure_pursuit_point: {pure_pursuit_point}")
            #     p.removeAllUserDebugItems()
            #     if self.pursuit_debug_point_id is not None:
            #         p.removeUserDebugItem(self.pursuit_debug_point_id, physicsClientId=self.client)
            #     self.pursuit_debug_point_id = p.addUserDebugPoints([[pure_pursuit_point[0], pure_pursuit_point[1], 0.0]],
            #                                                    [[0, 1, 0]], pointSize=10, physicsClientId=self.client)
            #     self.bicycle.draw_circle(center_pos=[obs[0], obs[1], 0.0], radius=self.center_radius, color=[1, 0, 0])

        delta_angle = center_angle - self.prev_center_angle
        reward = self._reward_fun(bicycle_obs, lidar_data=obs[9], is_collision=obs[10])

        # self.prev_center_angle = center_angle
        self.prev_dist_to_goal = distance_to_goal
        # self.prev_bicycle_pos = (obs[0], obs[1])
        # self.prev_yaw_angle = obs[2]

        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            self.truncated = True

        return ret_obs, reward, self.terminated, self.truncated, {"reward": reward}

    def reset(self, seed=None, options=None):
        self.terminated = False
        self.truncated = False
        self._elapsed_steps = 0

        self.pursuit_debug_point_id = None
        self.index = 0

        if self.gui:
            print(f"self.episode_rwd={self.episode_rwd}")
        self.episode_rwd = {"1": 0.0, "2": 0.0, "3": 0.0, "4": 0.0}

        self.goal = generate_target_position()
        if self.goal == None:
            print(f">>>[上层环境] 目标点生成失败，重置环境...")
            self.reset()
        goal = Goal(self.client, self.goal)
        if self.prev_goal_id is not None:
            p.removeBody(self.prev_goal_id, self.client)
        self.prev_goal_id = goal.id

        obs = self.bicycle.reset()

        distance_to_goal = np.linalg.norm(np.array([obs[0], obs[1]]) - np.array(self.goal))
        angle_to_goal = my_tools.calculate_goal_angle_gemini(obs[0], obs[1], obs[2], self.goal[0], self.goal[1])
        processed_lidar_data = self._process_lidar_data(lidar_data=obs[9])
        bicycle_obs = np.array([distance_to_goal, angle_to_goal, obs[2], obs[5]], dtype=np.float32)
        ret_obs = np.concatenate((processed_lidar_data, bicycle_obs))

        self.prev_dist_to_goal = distance_to_goal
        # self.prev_bicycle_pos = (obs[0], obs[1])
        # self.prev_center_angle = math.pi / 2
        # self.prev_yaw_angle = math.pi / 2

        return ret_obs, {}

    def _reward_fun(self, obs, lidar_data, is_collision=False):
        self.terminated = False
        self.truncated = False
        # obs 机器人与目标点距离, 机器人与目标点的角度, 翻滚角, 翻滚角角速度, 车把角度, 车把角速度
        distance_to_goal = obs[0]
        angle_to_target = obs[1]
        roll_angle = obs[2]

        # ========== 平衡奖励 ==========
        balance_rwd = 0.0
        if math.fabs(roll_angle) >= 0.45:
            self.terminated = True
            balance_rwd = -120.0
        # ========== 平衡奖励 ==========

        # ========== 导航奖励 ==========
        diff_dist = (self.prev_dist_to_goal - distance_to_goal) * 3.0
        distance_rwd = diff_dist if diff_dist > 0 else 1.2 * diff_dist
        angle_rwd = math.cos(angle_to_target) * 0.02

        goal_rwd = 0.0
        if distance_to_goal <= 10.0:
            if diff_dist > 0.0:
                distance_rwd += (0.3 * distance_rwd)
            else:
                distance_rwd += (0.1 * distance_rwd)
            angle_rwd = math.cos(angle_to_target) * 0.03
        elif distance_to_goal <= 5.0:
            angle_rwd = math.cos(angle_to_target) * 0.04
        elif distance_to_goal <= self.goal_threshold:
            print("=====到达目标点=====")
            self.terminated = True
            goal_rwd = 500.0

        navigation_rwd = distance_rwd + goal_rwd + angle_rwd
        # ========== 导航奖励 ==========

        # ========== 避障奖励 ==========
        # obstacle_penalty = 0.0
        # min_obstacle_dist = np.min(lidar_data)
        # if min_obstacle_dist <= self.safe_distance:
        #     obstacle_penalty = -10.0
        # avoid_obstacle_rwd = obstacle_penalty
        # ========== 避障奖励 ==========

        if self.gui:
            # print(f"distance_rwd: {distance_rwd:.5f}, "
            #       f"goal_rwd: {goal_rwd:.5f}, "
            #       f"angle_penalty: {angle_rwd:.5f}")
            self.episode_rwd["1"] += distance_rwd
            self.episode_rwd["2"] += angle_rwd

        # total_reward = balance_rwd + navigation_rwd + avoid_obstacle_rwd
        total_reward = balance_rwd + navigation_rwd

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

    def _process_lidar_data(self, lidar_data):
        distance_reshaped = lidar_data.reshape(60, 3)  # 使用reshape将其变为(60, 3)的形状，方便每3个元素进行平均
        averaged_distance = np.mean(distance_reshaped, axis=1, keepdims=True).flatten().tolist()  # 对每一行取平均值
        return np.array(averaged_distance, dtype=np.float32)

    def _calculate_new_position(self, robot_x, robot_y, yaw, distance, angle):
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
        # 计算总角度
        total_angle = yaw + angle
        # 将角度归一化到[-π, π]范围
        total_angle = (total_angle + math.pi) % (2 * math.pi) - math.pi
        # 计算新坐标
        new_x = robot_x + distance * math.cos(total_angle)
        new_y = robot_y + distance * math.sin(total_angle)

        return new_x, new_y


if __name__ == '__main__':
    env = gymnasium.make('BicycleDmzEnv-v0', gui=True)
    obs, _ = env.reset()
    # check_observation_space(obs, env.observation_space)
    for i in range(10000):
        action = np.array([-0.5], np.float32)
        obs, _, terminated, truncated, infos = env.step(action)

        # if terminated or truncated:
        #     obs, _ = env.reset()
        time.sleep(1. / 240.)

    env.close()
