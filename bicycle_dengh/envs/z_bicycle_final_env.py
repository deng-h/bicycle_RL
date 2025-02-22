import time
import gymnasium
import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from bicycle_dengh.resources.z_bicycle_navi import ZBicycleNavi
from bicycle_dengh.resources.z_bicycle_final import ZBicycleFinal
from bicycle_dengh.resources.goal import Goal
from playground.a_start.create_grid_map import create_grid_map2
from playground.a_start.create_obstacle import create_obstacle
from playground.a_start.create_obstacle import generate_target_position
from playground.a_start.a_star_algo import a_star_pathfinding
from playground.a_start.visualize_path import visualize_path
from playground.a_start.visualize_path import smooth_path_bezier

import math
from utils import my_tools
from simple_pid import PID
import yaml
import platform
from stable_baselines3 import PPO


def normalize_array_to_minus_one_to_one(arr, a, b):
    """
    将数组arr从区间[a, b]归一化到[-1, 1]

    参数:
    arr -- 要归一化的数组
    a -- 区间下限
    b -- 区间上限

    返回:
    归一化后的数组
    """
    if a.all() == b.all():
        raise ValueError("a 和 b 不能相等")

    m = 2 / (b - a)
    c = - (b + a) / (b - a)
    return m * arr + c


def calculate_angle_to_target(a, b, phi, x, y):
    """
    计算机器人与目标点之间的角度

    参数：
    a, b - 机器人的当前坐标 (a, b)
    phi - 机器人的当前偏航角，单位为弧度
    x, y - 目标点的坐标 (x, y)

    返回：
    机器人与目标点之间的角度，单位为弧度
    """
    # 计算目标点相对于机器人的方向
    delta_x = x - a
    delta_y = y - b

    # 计算目标方向的角度
    target_angle = math.atan2(delta_y, delta_x)

    # 计算机器人与目标点之间的相对角度
    angle_to_target = target_angle - phi

    # 将角度规范化到 [-π, π] 范围内
    angle_to_target = (angle_to_target + math.pi) % (2 * math.pi) - math.pi

    return angle_to_target


class ZBicycleFinalEnv(gymnasium.Env):
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, gui=False):
        system = platform.system()
        if system == "Windows":
            yaml_file_path = "D:\\data\\1-L\\9-bicycle\\bicycle-rl\\bicycle_dengh\envs\BicycleMazeLidarEnvConfig.yaml"
        else:
            yaml_file_path = "/home/chen/denghang/bicycle-rl/bicycle_dengh/envs/BicycleMazeLidarEnvConfig.yaml"
        with open(yaml_file_path, "r", encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.ppo_model = PPO.load("./ppo_model_omni_0607_1820.zip", device='cpu')
        self.goal = (1, 1)
        self.goal_temp = (-4, 2)
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
        self.episode_rwd = {"1": 0, "2": 0, "3": 0}
        self.radians_array = np.linspace(0, math.pi, 60)
        x = 40  # 想要取出的中间元素个数
        array_length = 60  # 总共60个元素
        self.start_index = (array_length - x) // 2
        self.end_index = self.start_index + x
        self.center_radius = 3.5
        self.prev_bicycle_pos = (-4, -1)
        self.prev_yaw_angle = 1.57
        self.prev_lower_obs = None
        self.prev_obs = None

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
        # 第二部分：[-50, -math.pi, -1.5, -20] ~ [50, math.pi, 1.5, 20]
        part2_low = np.array([0.0, -math.pi, -1.57, -15.0])
        part2_high = np.array([100.0, math.pi, 1.57, 15.0])
        # 合并两部分 low 和 high 数组
        low = np.concatenate([part1_low, part2_low])
        high = np.concatenate([part1_high, part2_high])

        # 60个距离数据, 机器人与目标点距离, 机器人与目标点的角度, 车把角度, 车把角速度
        self.observation_space = gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)

        # 机器人与目标点距离, 机器人与目标点的角度, 翻滚角, 翻滚角角速度, 车把角度, 车把角速度, 后轮速度, 飞轮速度
        self.lower_actual_observation_space = gymnasium.spaces.box.Box(
            low=np.array([0.0, -math.pi, -math.pi, -15.0, -1.57, -15.0, 0.0, -self.max_flywheel_vel]),
            high=np.array([100.0, math.pi, math.pi, 15.0, 1.57, 15.0, 10.0, self.max_flywheel_vel]),
            shape=(8,),
            dtype=np.float32)

        # action_space[车把角度，前后轮速度, 飞轮速度]
        self.lower_action_space = gymnasium.spaces.box.Box(
            low=np.array([-1.57, 0.0, -self.max_flywheel_vel]),
            high=np.array([1.57, 5.0, self.max_flywheel_vel]),
            shape=(3,),
            dtype=np.float32)
        self.lower_action_space_low, self.lower_action_space_high = self.lower_action_space.low, self.lower_action_space.high

        if self.gui:
            self.client = p.connect(p.GUI)
            self.camera_distance_param = p.addUserDebugParameter('camera_distance_param', 2, 60, 5)
            self.camera_yaw_param = p.addUserDebugParameter('camera_yaw_param', -180, 180, 0)
            self.camera_pitch_param = p.addUserDebugParameter('camera_pitch_param', -90, 90, -30)
            self.bicycle_vel_param = p.addUserDebugParameter('bicycle_vel_param', 0.0, 3.0, 1.0)
            self.handlebar_angle_param = p.addUserDebugParameter('handlebar_angle_param', -1.57, 1.57, 0)
            self.flywheel_param = p.addUserDebugParameter('flywheel_param', -40, 40, 0)
            # 设置俯视视角
            # p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-89,
            #                              cameraTargetPosition=[0, 12, 10])
        else:
            self.client = p.connect(p.DIRECT)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

        obstacle_ids = create_obstacle(self.client)
        self.bicycle = ZBicycleFinal(self.client, obstacle_ids=obstacle_ids)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        # p.changeDynamics(plane_id, -1, lateralFriction=1.5)  # 更改地面物体的动力学参数，包括摩擦系数，-1表示所有部件
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.setGravity(0, 0, -10, physicsClientId=self.client)
        # p.setRealTimeSimulation(0, physicsClientId=self.client)
        p.setTimeStep(1. / 24., self.client)

    def step(self, action):
        rescaled_action = self._rescale_action(action)
        center_angle = rescaled_action[0]
        self.pursuit_point = self._calculate_new_position(self.prev_bicycle_pos[0],
                                                          self.prev_bicycle_pos[1],
                                                          self.prev_yaw_angle,
                                                          self.center_radius,
                                                          center_angle)

        lower_obs = self._get_lower_obs(self.prev_obs, self.pursuit_point)
        lower_action, _ = self.ppo_model.predict(lower_obs, deterministic=True)
        lower_action = self._rescale_lower_action(lower_action)
        print(f"lower_action: {lower_action}")
        self.bicycle.apply_action(lower_action)
        p.stepSimulation(physicsClientId=self.client)

        obs = self.bicycle.get_observation()
        # 给上层网络的观测数据
        handbar_angle = obs[5]
        handbar_vel = obs[6]
        processed_lidar_data = self._process_lidar_data(lidar_data=obs[9])
        distance_to_goal = math.sqrt((self.goal[0] - obs[0]) ** 2 + (self.goal[1] - obs[1]) ** 2)
        angle_to_goal = calculate_angle_to_target(obs[0], obs[1], obs[2], self.goal[0], self.goal[1])
        bicycle_obs = np.array([distance_to_goal, angle_to_goal, handbar_angle, handbar_vel], dtype=np.float32)
        total_obs = np.concatenate((processed_lidar_data, bicycle_obs))

        # 给下层网络的观测数据
        self.prev_lower_obs = self._get_lower_obs(obs, self.pursuit_point)

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
                                  handbar_angle=handbar_angle, goal_angle=angle_to_goal, handbar_vel=handbar_vel,
                                  is_collided=obs[10], is_proximity=obs[11])

        self.prev_dist_to_goal = self.prev_lower_obs[0]
        self.prev_bicycle_pos = (obs[0], obs[1])
        self.prev_obs = obs

        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            formatted_dict = {key: "{:.8f}".format(value) for key, value in self.episode_rwd.items()}
            print(f">>>跑满啦！奖励值{formatted_dict}")
            self.truncated = True

        return total_obs, reward, self.terminated, self.truncated, {}

    def reset(self, seed=None, options=None):
        self.terminated = False
        self.truncated = False
        self._elapsed_steps = 0
        self.pursuit_point = (-4, -1)
        self.goal = generate_target_position()  # 全局目标点

        if self.goal == None:
            print(f">>>目标点生成失败，重置环境...")
            self.goal = (100, 100)
            self.reset()

        goal = Goal(self.client, self.goal)
        # 因为没有重置环境，每次reset后要清除先前的Goal
        if self.prev_goal_id is not None:
            p.removeBody(self.prev_goal_id, self.client)
        self.prev_goal_id = goal.id

        if self.gui:
            formatted_dict = {key: "{:.8f}".format(value) for key, value in self.episode_rwd.items()}
            print(f">>>episode_rwd: {formatted_dict}")
            print(f">>>前往目标点: ({self.goal[0]:.2F}, {self.goal[0]:.2F})")
        self.episode_rwd = {"1": 0, "2": 0, "3": 0}

        obs = self.bicycle.reset()

        # 给上层网络的观测数据
        handbar_angle = obs[5]
        handbar_vel = obs[6]
        processed_lidar_data = self._process_lidar_data(lidar_data=obs[9])
        distance_to_goal = math.sqrt((self.goal[0] - obs[0]) ** 2 + (self.goal[1] - obs[1]) ** 2)
        angle_to_goal = calculate_angle_to_target(obs[0], obs[1], obs[2], self.goal[0], self.goal[1])
        bicycle_obs = np.array([distance_to_goal, angle_to_goal, handbar_angle, handbar_vel], dtype=np.float32)
        total_obs = np.concatenate((processed_lidar_data, bicycle_obs))

        # 给下层网络的观测数据
        self.prev_lower_obs = self._get_lower_obs(obs, self.pursuit_point)

        self.prev_dist_to_goal = distance_to_goal
        self.prev_bicycle_pos = (-4, -1)
        self.prev_yaw_angle = 1.57
        self.prev_obs = obs

        return total_obs, {}

    def _get_lower_obs(self, obs, pursuit_point):
        distance_to_goal = math.sqrt((pursuit_point[0] - obs[0]) ** 2 + (pursuit_point[1] - obs[1]) ** 2)
        angle_to_target = calculate_angle_to_target(obs[0], obs[1], obs[2], pursuit_point[0], pursuit_point[1])
        lower_obs = [distance_to_goal, angle_to_target, obs[3], obs[4], obs[5], obs[6], obs[7], obs[8]]
        normalized_obs = normalize_array_to_minus_one_to_one(lower_obs, self.lower_actual_observation_space.low,
                                                             self.lower_actual_observation_space.high)
        normalized_obs = np.array(normalized_obs, dtype=np.float32)
        return normalized_obs

    def _reward_fun(self, obs, roll_angle, processed_lidar_data, handbar_angle, goal_angle, handbar_vel,
                    is_collided=False, is_proximity=False):
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
        approach_rwd = diff_dist if diff_dist > 0.0 else 1.2 * diff_dist

        if distance_to_goal <= 7.0:
            if approach_rwd > 0.0:
                approach_rwd *= 1.2
            else:
                approach_rwd *= 1.3

        # 转换到以自行车yaw为y轴正方向的坐标系下
        handbar_angle += (math.pi / 2.0)
        direction_rwd = math.cos(handbar_angle - goal_angle) * 0.01
        distance_rwd = -0.0008 * distance_to_goal

        goal_rwd = 0.0
        if distance_to_goal <= self.goal_threshold:
            formatted_dict = {key: "{:.8f}".format(value) for key, value in self.episode_rwd.items()}
            print(
                f">>>到达目标点({self.goal[0]:.2F}, {self.goal[1]:.2F})！存活{self._elapsed_steps}步，奖励值{formatted_dict}")
            goal_rwd = 250.0
            self.terminated = True
        navigation_rwd = approach_rwd + distance_rwd + direction_rwd + goal_rwd
        # ========== 导航奖励 ==========

        # ========== 避障奖励 ==========
        collision_penalty = 0.0
        if is_collided:
            collision_penalty = -20.0
            self.terminated = True
            if self.gui:
                formatted_dict = {key: "{:.8f}".format(value) for key, value in self.episode_rwd.items()}
                print(f">>>碰撞！存活{self._elapsed_steps}步，奖励值{formatted_dict}")

        # middle_elements = processed_lidar_data[self.start_index:self.end_index]  # 使用数组切片取出中间元素
        # # 离障碍物一定范围内开始做惩罚
        # min_val = np.min(middle_elements)
        # obstacle_penalty = 0.0
        # if min_val <= 3.5:
        #     obstacle_penalty = max(0.1, min_val)
        #     obstacle_penalty = -1.0 / (obstacle_penalty * 50)
        # obstacle_penalty = 0.0
        # indices_less_than_4  = np.where(processed_lidar_data < 4.0)[0]  # 找出距离小于n的元素的索引
        # indices_great_than_4 = np.where(processed_lidar_data > 4.0)[0]

        # if len(indices_great_than_4) > 0:
        #     no_obstacle_direction = np.array(self.radians_array[indices_great_than_4])
        #     absolute_differences = np.abs(no_obstacle_direction - goal_angle)
        #     min_absolute_difference = np.min(absolute_differences)
        #     if min_absolute_difference < 0.45 and np.abs(handbar_angle - goal_angle) < 0.43:
        #         direction_rwd = 0.02

        # turn_rwd = 0.0  # 车把打角奖励，目的是不要朝着障碍物方向走
        # if len(indices_less_than_4) > 0:
        #     obstacle_direction = np.array(self.radians_array[indices_less_than_4])
        #     absolute_differences = np.abs(obstacle_direction - handbar_angle)
        #     min_absolute_difference = np.min(absolute_differences)
        #     # print("数组 processed_lidar_data 中数值小于 5 的元素的索引:", indices_less_than_5)
        #     # print("数组 obstacle_direction 中元素与 handbar_angle 做差取绝对值的结果:", absolute_differences)
        #     # print("根据索引从 radians_array 中提取的角度数组 obstacle_direction:", obstacle_direction)
        #     # print(f"最小的值:{min_absolute_difference}, handbar_angle:{handbar_angle}" )
        #     if min_absolute_difference < 0.17:
        #         turn_rwd = -0.001k

        # 太靠近给固定惩罚
        proximity_penalty = -0.02 if is_proximity else 0.0

        # ds_rwd = self._deepseek_design_rwd(bicycle_yaw, processed_lidar_data, self.radians_array) / 100000.0

        avoid_obstacle_rwd = collision_penalty + proximity_penalty
        # ========== 避障奖励 ==========

        # 车把速度限制
        handbar_vel_rwd = -0.5 if np.abs(handbar_vel) > 7.0 else 0.0

        # if self.gui:
        #     print(f"approach_rwd: {approach_rwd}, distance_rwd: {distance_rwd}, "
        #             f"proximity_penalty: {proximity_penalty}")

        self.episode_rwd["1"] += approach_rwd
        self.episode_rwd["2"] += distance_rwd
        # self.episode_rwd["3"] += ds_rwd

        total_reward = balance_rwd + navigation_rwd + avoid_obstacle_rwd + handbar_vel_rwd

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

    def _rescale_lower_action(self, scaled_action):
        return self.lower_action_space_low + (0.5 * (scaled_action + 1.0) * (self.lower_action_space_high - self.action_low))

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
    env = gymnasium.make('ZBicycleFinalEnv-v0', gui=True)
    obs, _ = env.reset()
    # check_observation_space(obs, env.observation_space)
    for i in range(10000):
        action = np.array([0.0], np.float32)
        obs, _, terminated, truncated, infos = env.step(action)
        # time.sleep(0.1)
        if terminated or truncated:
            obs, _ = env.reset()
        time.sleep(1. / 24.)

    env.close()

# 记录
# 2/22  角度奖励用cos函数
#       圆柱障碍物半径从1.25改为1.0
#       优化了log样式
#       学习率改为4e-4，增加epochs
#       先从简单的障碍物环境开始训练
