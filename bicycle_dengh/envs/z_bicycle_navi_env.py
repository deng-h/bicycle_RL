import time
import gymnasium
import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from bicycle_dengh.resources.z_bicycle_navi import ZBicycleNavi
from bicycle_dengh.resources.goal import Goal
from playground.a_start.create_grid_map import create_grid_map2
from playground.a_start.create_obstacle import create_obstacle
from playground.a_start.a_star_algo import a_star_pathfinding
from playground.a_start.visualize_path import visualize_path
from playground.a_start.visualize_path import smooth_path_bezier
from playground.a_start.get_goal_pos import get_goal_pos

import math
from utils import my_tools
from simple_pid import PID
import yaml
import platform
import random


class ZBicycleNaviEnv(gymnasium.Env):
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, gui=False):
        system = platform.system()
        if system == "Windows":
            yaml_file_path = "D:\\data\\1-L\\9-bicycle\\bicycle-rl\\bicycle_dengh\envs\BicycleMazeLidarEnvConfig.yaml"
        else:
            yaml_file_path = "/home/chen/denghang/bicycle-rl/bicycle_dengh/envs/BicycleMazeLidarEnvConfig.yaml"
        with open(yaml_file_path, "r", encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.goal = (1, 1)
        self.terminated = False
        self.truncated = False
        self.gui = gui
        self.max_flywheel_vel = 120.
        self.prev_goal_id = None
        self.prev_dist_to_goal = 0.
        self.roll_angle_pid = PID(1100, 0, 0, setpoint=0.0)
        self.current_roll_angle = 0.0
        self._max_episode_steps = self.config["max_episode_steps"]
        self.goal_threshold = self.config["goal_threshold"]
        self.safe_distance = self.config["safe_distance"]
        self._elapsed_steps = None
        self.bicycle_start_pos = (1, 1)
        self.action_space = gymnasium.spaces.box.Box(low=-1., high=1., shape=(1,), dtype=np.float32)
        self.episode_rwd = {"1": 0, "2": 0, "3": 0, "4": 0}
        self.radians_array = np.linspace(0, math.pi, 60)
        x = 40  # 想要取出的中间元素个数
        array_length = 60  # 总共60个元素
        self.start_index = (array_length - x) // 2
        self.end_index = self.start_index + x

        # 车把角度
        self.actual_action_space = gymnasium.spaces.box.Box(
            low=np.array([-1.5]),
            high=np.array([1.5]),
            shape=(1,),
            dtype=np.float32)

        self.action_low, self.action_high = self.actual_action_space.low, self.actual_action_space.high

        # 第一部分：60 个 [-12.0] ~ [12.0]
        num_groups = 60
        group1_low = np.array([-12.0])
        group1_high = np.array([12.0])
        # 使用 numpy.tile 快速创建重复的 low 和 high 数组
        part1_low = np.tile(group1_low, num_groups)
        part1_high = np.tile(group1_high, num_groups)
        # 第二部分：[-50, -math.pi, -1.5, -20] ~ [50, math.pi, 1.5, 20]
        part2_low = np.array([-50.0, -math.pi, -1.5, -20.0])
        part2_high = np.array([50.0, math.pi, 1.5, 20.0])
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
            # 设置俯视视角
            p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-89,
                                         cameraTargetPosition=[0, 12, 10])
        else:
            self.client = p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

        obstacle_ids = create_obstacle(self.client)
        self.bicycle = ZBicycleNavi(self.client, self.max_flywheel_vel, obstacle_ids=obstacle_ids)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.changeDynamics(plane_id, -1, lateralFriction=1.5)  # 更改地面物体的动力学参数，包括摩擦系数，-1表示所有部件
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.setGravity(0, 0, -10, physicsClientId=self.client)
        # p.setRealTimeSimulation(0, physicsClientId=self.client)
        p.setTimeStep(1. / 90., self.client)

    def step(self, action):
        rescaled_action = self._rescale_action(action)
        self.bicycle.apply_action2(rescaled_action)
        p.stepSimulation(physicsClientId=self.client)
        obs = self.bicycle.get_observation()
        distance_to_goal_temp = np.linalg.norm(np.array([obs[0], obs[1]]) - np.array(self.goal))
        angle_to_target = my_tools.calculate_goal_angle_gemini(obs[0], obs[1], obs[2], self.goal[0], self.goal[1])
        bicycle_obs = np.array([distance_to_goal_temp, angle_to_target, obs[4], obs[5]], dtype=np.float32)

        processed_lidar_data = self._process_lidar_data(lidar_data=obs[6])

        total_obs = np.concatenate((processed_lidar_data, bicycle_obs))

        if self.gui:
            keys = p.getKeyboardEvents()
            if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
                time.sleep(10000)
            elif ord('z') in keys and keys[ord('z')] & p.KEY_WAS_TRIGGERED:
                # 设置俯视视角
                p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-89,
                                             cameraTargetPosition=[0, 12, 10])
            elif ord('x') in keys and keys[ord('x')] & p.KEY_WAS_TRIGGERED:
                bike_pos, _ = p.getBasePositionAndOrientation(self.bicycle.bicycleId, physicsClientId=self.client)
                camera_distance = p.readUserDebugParameter(self.camera_distance_param)
                camera_yaw = p.readUserDebugParameter(self.camera_yaw_param)
                camera_pitch = p.readUserDebugParameter(self.camera_pitch_param)
                p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, bike_pos)

        reward = self._reward_fun(bicycle_obs, roll_angle=obs[3], processed_lidar_data=processed_lidar_data,
                                  handbar_angle=obs[4], goal_angle=angle_to_target,is_collided=obs[7], is_proximity=obs[8])

        self.prev_dist_to_goal = distance_to_goal_temp

        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            print(f">>>跑满啦！奖励值{self.episode_rwd}")
            self.truncated = True

        return total_obs, reward, self.terminated, self.truncated, {}

    def reset(self, seed=None, options=None):
        self.terminated = False
        self.truncated = False
        self._elapsed_steps = 0
        self.goal = get_goal_pos()
        goal = Goal(self.client, self.goal)
        # 因为没有重置环境，每次reset后要清除先前的Goal
        if self.prev_goal_id is not None:
            p.removeBody(self.prev_goal_id, self.client)
        self.prev_goal_id = goal.id
        if self.gui:
            print(f">>>episode_rwd: {self.episode_rwd}")
            print(f">>>前往目标点: {self.goal}")
        self.episode_rwd = {"1": 0, "2": 0, "3": 0, "4": 0}
        obs = self.bicycle.reset()
        processed_lidar_data = self._process_lidar_data(lidar_data=obs[6])

        distance_to_goal_temp = np.linalg.norm(np.array([obs[0], obs[1]]) - np.array(self.goal))
        angle_to_target = my_tools.calculate_goal_angle_gemini(obs[0], obs[1], obs[2], self.goal[0], self.goal[1])
        bicycle_obs = np.array([distance_to_goal_temp, angle_to_target, obs[4], obs[5]], dtype=np.float32)
        self.prev_dist_to_goal = distance_to_goal_temp
        total_obs = np.concatenate((processed_lidar_data, bicycle_obs))
        return total_obs, {}

    def _reward_fun(self, obs, roll_angle, processed_lidar_data, handbar_angle, goal_angle, is_collided=False, is_proximity=False):
        self.terminated = False
        self.truncated = False
        # action 车把角度
        # obs 机器人与目标点距离, 机器人与目标点的角度, 翻滚角, 车把角度, 车把角速度
        distance_to_goal = obs[0]

        # ========== 平衡奖励 ==========
        balance_rwd = 0.0
        if math.fabs(roll_angle) >= 0.35:
            self.terminated = True
            balance_rwd = -100.0
        # ========== 平衡奖励 ==========

        # ========== 导航奖励 ==========
        diff_dist = (self.prev_dist_to_goal - distance_to_goal) * 10.0
        distance_rwd = diff_dist if diff_dist > 0.0 else 1.3 * diff_dist

        if distance_to_goal <= 7.0:
            if distance_rwd > 0.0:
                distance_rwd *= 1.2
            else:
                distance_rwd *= 1.5

        goal_rwd = 0.0
        if distance_to_goal <= self.goal_threshold:
            print(f">>>到达目标点({self.goal[0]}, {self.goal[1]})！存活{self._elapsed_steps}步，奖励值{self.episode_rwd}")
            goal_rwd = 200.0
            self.terminated = True
        navigation_rwd = distance_rwd + goal_rwd
        # ========== 导航奖励 ==========

        # ========== 避障奖励 ==========
        collision_penalty = 0.0
        if is_collided:
            collision_penalty = -100.0
            self.terminated = True
            if self.gui:
                print(f">>>碰撞！存活{self._elapsed_steps}步，奖励值{self.episode_rwd}")

        # middle_elements = processed_lidar_data[self.start_index:self.end_index]  # 使用数组切片取出中间元素
        # # 离障碍物一定范围内开始做惩罚
        # min_val = np.min(middle_elements)
        # obstacle_penalty = 0.0
        # if min_val <= 3.5:
        #     obstacle_penalty = max(0.1, min_val)
        #     obstacle_penalty = -1.0 / (obstacle_penalty * 50)
        # obstacle_penalty = 0.0
        indices_less_than_4  = np.where(processed_lidar_data < 4.0)[0]  # 找出距离小于n的元素的索引
        indices_great_than_4 = np.where(processed_lidar_data > 4.0)[0]
        # 转换到以自行车yaw为y轴正方向的坐标系下
        handbar_angle += (math.pi / 2.0)
        direction_rwd = 0.0  # 如果目标点位于激光雷达的空旷范围内 并且 车头朝向目标点 加分
        if len(indices_great_than_4) > 0:
            no_obstacle_direction = np.array(self.radians_array[indices_great_than_4])
            absolute_differences = np.abs(no_obstacle_direction - goal_angle)
            min_absolute_difference = np.min(absolute_differences)
            if min_absolute_difference < 0.45 and np.abs(handbar_angle - goal_angle) < 0.43:
                direction_rwd = 0.02

        turn_rwd = 0.0  # 车把打角奖励，目的是不要朝着障碍物方向走
        if len(indices_less_than_4) > 0:
            obstacle_direction = np.array(self.radians_array[indices_less_than_4])
            absolute_differences = np.abs(obstacle_direction - handbar_angle)
            min_absolute_difference = np.min(absolute_differences)
            # print("数组 processed_lidar_data 中数值小于 5 的元素的索引:", indices_less_than_5)
            # print("数组 obstacle_direction 中元素与 handbar_angle 做差取绝对值的结果:", absolute_differences)
            # print("根据索引从 radians_array 中提取的角度数组 obstacle_direction:", obstacle_direction)
            # print(f"最小的值:{min_absolute_difference}, handbar_angle:{handbar_angle}" )
            if min_absolute_difference < 0.17:
                turn_rwd = -0.001

        # 太靠近给固定惩罚
        proximity_penalty = 0.0
        if is_proximity:
            proximity_penalty = -0.04

        # ds_rwd = self._deepseek_design_rwd(bicycle_yaw, processed_lidar_data, self.radians_array) / 100000.0

        avoid_obstacle_rwd = collision_penalty + proximity_penalty + turn_rwd + direction_rwd
        # ========== 避障奖励 ==========

        # if self.gui:
        #     print(f"distance_rwd: {distance_rwd}, obstacle_penalty: {turn_rwd}, ds_rwd: {turn_rwd}, "
        #             f"proximity_penalty: {proximity_penalty}")
            
        self.episode_rwd["1"] += distance_rwd
        self.episode_rwd["2"] += turn_rwd
        # self.episode_rwd["3"] += ds_rwd

        total_reward = balance_rwd + navigation_rwd + avoid_obstacle_rwd

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

    def _deepseek_design_rwd(self, yaw, processed_lidar, theta_array):
        """
        计算基于当前yaw和雷达数据的奖励值。

        参数:
        yaw (float): 机器人的当前朝向（弧度），范围0到2π。
        processed_lidar (np.array): 处理后的60×1雷达距离数据。
        theta_array (np.array): 60×1的弧度数组，范围0到π，对应雷达的各个角度。

        返回:
        float: 计算出的奖励值。
        """
        sum_x = 0.0
        sum_y = 0.0

        for i in range(60):
            theta_i = theta_array[i]
            distance_i = processed_lidar[i]
            global_angle = yaw + (theta_i - math.pi / 2)  # 将雷达角度转换为全局坐标系
            # 计算向量分量并累加
            sum_x += distance_i * math.cos(global_angle)
            sum_y += distance_i * math.sin(global_angle)

        # 计算合成向量的模和方向
        magnitude = math.hypot(sum_x, sum_y)
        if magnitude == 0:
            return 0.0  # 所有距离均为0时返回0

        desired_direction = math.atan2(sum_y, sum_x)

        # 计算yaw与目标方向的差异
        angle_diff = yaw - desired_direction
        # 将角度差规范到[-π, π]范围内
        angle_diff = (angle_diff + math.pi) % (2 * math.pi) - math.pi

        # 奖励为模长乘以余弦相似度
        reward = magnitude * math.cos(angle_diff)

        return reward


def check_observation_space(observation, observation_space):
    """
    检查 step() 返回的 observation 是否在 observation_space 范围内。
    """
    errors = []
    for key, space in observation_space.spaces.items():
        if isinstance(space, gymnasium.spaces.Box):
            obs = observation[key]
            # 检查是否超出范围
            low_violation = obs < space.low
            high_violation = obs > space.high

            # 如果存在超出范围的值，记录下来
            if np.any(low_violation) or np.any(high_violation):
                errors.append({
                    "key": key,
                    "out_of_bounds_indices": np.where(low_violation | high_violation)[0],
                    "actual_values": obs,
                    "low_bound": space.low,
                    "high_bound": space.high,
                })

    if errors:
        print("Observation out of bounds:")
        for error in errors:
            print(f"Key: {error['key']}")
            for idx in error['out_of_bounds_indices']:
                print(f"  Index {idx}: value={error['actual_values'][idx]}, "
                      f"low={error['low_bound'][idx]}, high={error['high_bound'][idx]}")
    else:
        print("All observations are within bounds!")


if __name__ == '__main__':
    env = gymnasium.make('ZBicycleNaviEnv-v0', gui=True)
    obs, _ = env.reset()
    # check_observation_space(obs, env.observation_space)
    for i in range(10000):
        action = np.array([0.0], np.float32)
        obs, _, terminated, truncated, infos = env.step(action)

        # if terminated or truncated:
        #     obs, _ = env.reset()
        time.sleep(1. / 120.)

    env.close()
