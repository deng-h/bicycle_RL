import time
import gymnasium
import numpy as np
import pybullet_data
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from bicycle_dengh.resources.z_bicycle_final import ZBicycleFinal
from playground.normalize_action import NormalizeAction
import pybullet as p
import math
from playground.a_start.create_obstacle import generate_target_position
import yaml
import platform
from stable_baselines3 import PPO


class BicycleFinalEnv(gymnasium.Env):
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, gui=False):
        system = platform.system()
        if system == "Windows":
            yaml_file_path = "D:\\data\\1-L\\9-bicycle\\bicycle-rl\\bicycle_dengh\envs\BicycleMazeLidarEnvConfig.yaml"
        else:
            yaml_file_path = "/home/chen/denghang/bicycle-rl/bicycle_dengh/envs/BicycleMazeLidarEnvConfig.yaml"
        with open(yaml_file_path, "r", encoding='utf-8') as f:
            self.config = yaml.safe_load(f)

        self.lower_env = gymnasium.make('BicycleDenghEnvCopy-v0', gui=gui)
        self.lower_env = NormalizeAction(self.lower_env)
        self.lower_env_client = self.lower_env.client
        self.ppo_model = PPO.load("bicycle_dengh/envs/ppo_model_omni_0607_1820.zip", env=self.lower_env, device='cpu')
        # self.ppo_model = PPO.load("./ppo_model_omni_0607_1820.zip", env=self.lower_env, device='cpu')

        self.goal = (1, 1)
        self.terminated = False
        self.truncated = False
        self.gui = gui
        self.max_flywheel_vel = 120.
        self.prev_goal_id = None
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
        self.center_radius = 4.0
        self.prev_bicycle_pos = (-4, -1)
        self.prev_yaw_angle = 1.57
        self.prev_lower_obs = None
        self.prev_obs = None
        self.lower_obs = None
        self.obs_for_bicycle = None
        self.obs_for_navi = None
        self.prev_dist_to_goal = None
        self.last_turn_angle = None
        self.reset_flg = False

        # 上层网络的引导角度（弧度）
        self.actual_action_space = gymnasium.spaces.box.Box(
            low=np.array([-1.2]),
            high=np.array([1.2]),
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
        # 第二部分：离目标点距离、离目标点角度、车身偏航角、车把角度、车把角速度
        part2_low = np.array([-50.0, -math.pi, -math.pi, -1.57, -20.0])
        part2_high = np.array([50.0, math.pi, math.pi, 1.57, 20.0])
        # 合并两部分 low 和 high 数组
        low = np.concatenate([part1_low, part2_low])
        high = np.concatenate([part1_high, part2_high])

        # 60个距离数据, 机器人与目标点距离, 机器人与目标点的角度, 车把角度, 车把角速度
        self.observation_space = gymnasium.spaces.Box(low=low, high=high, dtype=np.float32)

    def step(self, action):
        if self.reset_flg:
            self.reset_flg = False
            action = np.array([0.0], np.float32)

        # 控制下层环境
        lower_action, _ = self.ppo_model.predict(self.obs_for_bicycle, deterministic=True)
        lower_obs, _, _, _, info = self.lower_env.step(lower_action)

        # 上层的action
        rescaled_action = self._rescale_action(action)
        turn_angle = rescaled_action[0]
        # 传递pursuit_point给下层的self.lower_env 如果自行车到达了跟踪点，才给下一个点
        if info['reached_goal']:
            self.pursuit_point = self._calculate_new_position(self.prev_bicycle_pos[0],
                                                              self.prev_bicycle_pos[1],
                                                              self.prev_yaw_angle,
                                                              self.center_radius,
                                                              turn_angle)
            # print(f">>>[上层环境] 到达子目标点！更新点为{self.pursuit_point}")
            self.lower_env.set_pursuit_point(self.pursuit_point)

        self.obs_for_bicycle = lower_obs
        self.obs_for_navi = info['for_navi_obs']

        reward = self._reward_fun(distance_to_goal=self.obs_for_navi[-5], turn_angle=turn_angle,
                                  processed_lidar_data=self.obs_for_navi[:60], is_fall_down=info['fall_down'],
                                  is_collided=info['is_collided'], is_proximity=info['is_proximity'])

        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            formatted_dict = {key: "{:.8f}".format(value) for key, value in self.episode_rwd.items()}
            print(f">>>[上层环境] 跑满啦！奖励值{formatted_dict}")
            self.truncated = True

        self.last_turn_angle = turn_angle
        self.prev_dist_to_goal = self.obs_for_navi[-5]
        self.prev_yaw_angle = self.obs_for_navi[-3]
        self.prev_bicycle_pos = (info['bicycle_x'], info['bicycle_y'])

        return self.obs_for_navi, reward, self.terminated, self.truncated, {"reached_goal": info['reached_goal']}

    def reset(self, seed=None, options=None):
        self.terminated = False
        self.truncated = False
        self.reset_flg = True
        self._elapsed_steps = 0
        self.goal = generate_target_position()  # 全局目标点

        if self.goal == None:
            print(f">>>[上层环境] 目标点生成失败，重置环境...")
            self.goal = (100, 100)
            self.reset()
            self.lower_env.reset(pursuit_point=(100, 100), final_goal=self.goal)

        self.pursuit_point = (-4, 1)
        self.lower_env.set_pursuit_point(self.pursuit_point)

        self.obs_for_bicycle, info = self.lower_env.reset(pursuit_point=self.pursuit_point, final_goal=self.goal)
        self.obs_for_navi = info['for_navi_obs']

        self.prev_dist_to_goal = self.obs_for_navi[-5]
        self.last_turn_angle = 0.0
        self.prev_yaw_angle = self.obs_for_navi[-3]
        self.prev_bicycle_pos = (info['bicycle_x'], info['bicycle_y'])

        self.episode_rwd = {"1": 0, "2": 0, "3": 0}

        return self.obs_for_navi, {}

    def _reward_fun(self, distance_to_goal, turn_angle, processed_lidar_data, is_fall_down=False, is_collided=False, is_proximity=False):
        self.terminated = False
        self.truncated = False

        goal_rwd = 0.0
        if distance_to_goal < self.goal_threshold:
            self.truncated = True
            goal_rwd = 100.0
            formatted_dict = {key: "{:.8f}".format(value) for key, value in self.episode_rwd.items()}
            print(f">>>[上层环境] 到达目标点({self.goal[0]:.2F}, {self.goal[1]:.2F})！存活{self._elapsed_steps}步，奖励值{formatted_dict}")

        collision_penalty = 0.0
        if is_collided:
            collision_penalty = -20.0
            self.terminated = True
            # formatted_dict = {key: "{:.8f}".format(value) for key, value in self.episode_rwd.items()}
            # print(f">>>[上层环境] 碰撞！存活{self._elapsed_steps}步，奖励值{formatted_dict}")

        # 距离目标点奖励
        diff_dist_to_goal = self.prev_dist_to_goal - distance_to_goal
        distance_rwd = diff_dist_to_goal / (5.0 / 24.0)
        if diff_dist_to_goal > 0.0:
            distance_rwd = (1.0 / 10.0) * distance_rwd
        else:
            distance_rwd = (1.2 / 10.0) * distance_rwd

        fall_down_penalty = 0.0
        if is_fall_down:
            formatted_dict = {key: "{:.8f}".format(value) for key, value in self.episode_rwd.items()}
            print(f">>>[上层环境] 摔倒！存活{self._elapsed_steps}步，奖励值{formatted_dict}")
            fall_down_penalty = -20.0
            self.terminated = True

        # turn_penalty = math.cos(self.last_turn_angle - turn_angle) * 0.01
        # print(f"turn_angle: {turn_angle}, handlebar_vel: {handlebar_vel}")

        middle_elements = processed_lidar_data[self.start_index:self.end_index]  # 使用数组切片取出中间元素
        # 离障碍物一定范围内开始做惩罚
        min_val = np.min(middle_elements)
        obstacle_penalty = 0.0
        if min_val <= 0.5:
            obstacle_penalty = max(0.1, min_val)
            obstacle_penalty = -1.0 / (obstacle_penalty * 50)
        elif min_val <= 1.0:
            obstacle_penalty = max(0.1, min_val)
            obstacle_penalty = -1.0 / (obstacle_penalty * 50)
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
        #         turn_rwd = -0.001

        # 太靠近给固定惩罚
        proximity_penalty = -0.02 if is_proximity else 0.0

        avoid_obstacle_rwd = collision_penalty + proximity_penalty

        # print(f"distance_rwd: {distance_rwd}, "
        #         f"turn_penalty: {turn_penalty}")

        self.episode_rwd["1"] += distance_rwd

        total_reward = avoid_obstacle_rwd + goal_rwd + distance_rwd + fall_down_penalty + obstacle_penalty

        return total_reward

    def render(self):
        pass

    def close(self):
        pass

    def _rescale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high] (no need for symmetric action space)
        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        return self.action_low + (0.5 * (scaled_action + 1.0) * (self.action_high - self.action_low))

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
    env = gymnasium.make('BicycleFinalEnv-v0', gui=True)
    obs, _ = env.reset()
    angle_array = [-1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
    # check_observation_space(obs, env.observation_space)
    index = 0
    angle = angle_array[index]
    for i in range(10000):
        action = np.array([0], np.float32)
        obs, _, terminated, truncated, infos = env.step(action)
        if infos["reached_goal"]:
            index += 1
            if index >= len(angle_array):
                index = 0
            angle = angle_array[index]
            # print(">>>[上层环境] 到达子目标点！")

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
