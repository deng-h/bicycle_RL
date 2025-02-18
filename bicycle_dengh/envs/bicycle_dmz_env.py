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
            yaml_file_path = "/root/bicycle-rl/bicycle_dengh/envs/icycleMazeLidarEnvConfig.yaml"
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
        self._max_episode_steps = self.config["max_episode_steps"]
        self.goal_threshold = self.config["goal_threshold"]
        self.safe_distance = self.config["safe_distance"]
        self._elapsed_steps = None
        self.center_radius = 5.0
        self.last_center_angle = 0.0
        self.action_space = gymnasium.spaces.box.Box(low=-1., high=1., shape=(1,), dtype=np.float32)

        # 飞轮速度, 以自行车为中心的角度
        self.actual_action_space = gymnasium.spaces.box.Box(
            low=np.array([-math.pi]),
            high=np.array([math.pi]),
            shape=(1,),
            dtype=np.float32)

        self.action_low, self.action_high = self.actual_action_space.low, self.actual_action_space.high

        # 机器人与目标点距离, 机器人与目标点的角度, 翻滚角, 翻滚角角速度, 车把角度, 车把角速度, 后轮速度, 飞轮速度
        self.observation_space = gymnasium.spaces.Dict({
            "lidar": gymnasium.spaces.box.Box(low=0., high=50., shape=(120,), dtype=np.float32),
            "bicycle": gymnasium.spaces.box.Box(
                low=np.array([0.0, -math.pi, -math.pi, -15.0, -1.57, -15.0, 0.0, -self.max_flywheel_vel]),
                high=np.array([100.0, math.pi, math.pi, 15.0, 1.57, 15.0, 10.0, self.max_flywheel_vel]),
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
        p.setRealTimeSimulation(0)
        p.setTimeStep(1. / 120., self.client)

    def step(self, action):
        rescaled_action = self._rescale_action(action)
        # fly_wheel_action = rescaled_action[0]
        center_angle = rescaled_action[0]
        # print(f"center_angle: {center_angle:.3f}")
        pure_pursuit_point = self._calculate_new_position(self.prev_bicycle_pos[0],
                                                          self.prev_bicycle_pos[1],
                                                          self.center_radius, center_angle)
        # print(f"pure_pursuit_point: {pure_pursuit_point:.3f}")
        p.addUserDebugPoints([[pure_pursuit_point[0], pure_pursuit_point[1], 0.55]], [[0, 0, 1]], pointSize=10, physicsClientId=self.client)
        # print(f"自行车坐标: {self.prev_bicycle_pos}, center_angle: {center_angle}, pure_pursuit_point: {pure_pursuit_point}")
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

        delta_angle = center_angle - self.last_center_angle
        self.last_center_angle = center_angle
        reward = self._reward_fun(bicycle_obs, lidar_data=obs[9], delta_angle=delta_angle, is_collision=obs[10])

        self.prev_dist_to_goal = distance_to_goal
        self.prev_bicycle_pos = (obs[0], obs[1])

        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            self.truncated = True

        return {"lidar": obs[9], "bicycle": bicycle_obs}, reward, self.terminated, self.truncated, {}

    def reset(self, seed=None, options=None):
        self.terminated = False
        self.truncated = False
        self._elapsed_steps = 0
        self.last_center_angle = 0.0
        x = random.randint(-25, 25)
        y = random.randint(5, 50)
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

        return {"lidar": obs[9], "bicycle": bicycle_obs}, {}

    def _reward_fun(self, obs, lidar_data, delta_angle, is_collision=False):
        self.terminated = False
        self.truncated = False
        # action 车把角度，前后轮速度, 飞轮速度, 相对路径点的距离x, 相对路径点的距离y
        # obs 机器人与目标点距离, 机器人与目标点的角度, 翻滚角, 翻滚角角速度, 车把角度, 车把角速度, 后轮速度, 飞轮速度
        distance_to_goal = obs[0]
        roll_angle = obs[2]
        bicycle_vel = obs[6]

        # ========== 平衡奖励 ==========
        balance_rwd = 0.0
        if math.fabs(roll_angle) >= 0.35:
            self.terminated = True
            balance_rwd = -10.0
        # else:
        #     balance_rwd = 1.0 - (math.fabs(roll_angle) / 0.35) * 2.0
        #     balance_rwd = max(-1.0, min(1.0, balance_rwd))
        # ========== 平衡奖励 ==========

        # ========== 导航奖励 ==========
        diff_dist = (self.prev_dist_to_goal - distance_to_goal) * 100.0
        distance_rwd = diff_dist if diff_dist > 0 else 1.5 * diff_dist
        goal_rwd = 0.0
        if distance_to_goal <= self.goal_threshold:
            goal_rwd = 100.0
        navigation_rwd = distance_rwd + goal_rwd
        # ========== 导航奖励 ==========

        # ========== 避障奖励 ==========
        obstacle_penalty = 0.0
        min_obstacle_dist = np.min(lidar_data)
        if min_obstacle_dist <= self.safe_distance:
            obstacle_penalty = -10.0
        avoid_obstacle_rwd = obstacle_penalty
        # ========== 避障奖励 ==========

        still_penalty = 0.0
        if math.fabs(bicycle_vel) <= 0.2:
            still_penalty = -1.0

        angle_penalty = math.fabs(delta_angle) * 0.5

        # print(f"distance_rwd: {distance_rwd:.3f}, "
        #       f"goal_rwd: {goal_rwd:.3f}, "
        #       f"angle_penalty: {angle_penalty:.3f}")

        # total_reward = balance_rwd + navigation_rwd + avoid_obstacle_rwd + still_penalty
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

    def _calculate_new_position(self, x, y, distance, angle):
        """
        根据当前位置、距离和角度计算新的位置。

        参数:
            x (float): 当前机器人的 x 坐标。
            y (float): 当前机器人的 y 坐标。
            distance (float): 随机生成的距离。
            angle (float): 随机生成的角度（弧度制，范围 -π 到 π）。

        返回:
            tuple: 新的机器人位置 (new_x, new_y)。
        """
        # 使用三角函数计算新位置
        delta_x = distance * math.cos(angle)
        delta_y = distance * math.sin(angle)

        # 计算新坐标
        new_x = x + delta_x
        new_y = y + delta_y

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
