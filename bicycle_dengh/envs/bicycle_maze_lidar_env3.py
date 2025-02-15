import time
import gymnasium
import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from bicycle_dengh.resources.bicycle_lidar import BicycleLidar
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



"""
想法：把自行车的翻滚角控制交给PID控制器，其他控制交给RL模型
"""
class BicycleMazeLidarEnv3(gymnasium.Env):
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, gui=False):
        system = platform.system()
        if system == "Windows":
            yaml_file_path = "D:\\data\\1-L\\9-bicycle\\bicycle-rl\\bicycle_dengh\envs\BicycleMazeLidarEnv3Config.yaml"
        else:
            yaml_file_path = "/root/bicycle-rl/bicycle_dengh/envs/icycleMazeLidarEnv3Config.yaml"
        with open(yaml_file_path, "r", encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        self.goal = (0, 0)
        self.goal_temp = (0, 0)
        self.terminated = False
        self.truncated = False
        self.gui = gui
        self.max_flywheel_vel = 120.
        self.prev_goal_id = None
        self.prev_dist_to_goal = 0.
        self.roll_angle_pid = PID(1100, 0, 0, setpoint=0.0)
        self.current_roll_angle = 0.0
        self._max_episode_steps = self.config["max_episode_steps"]
        self.proximity_threshold = self.config["proximity_threshold"]
        self.goal_threshold = self.config["goal_threshold"]
        self.safe_distance = self.config["safe_distance"]
        self.front_safe = self.config["front_safe"]
        self._elapsed_steps = None
        self.last_obs = None
        self.last_last_obs = None
        self.path_index = 0
        self.bicycle_start_pos = (1, 1)
        self.action_space = gymnasium.spaces.box.Box(low=-1., high=1., shape=(2,), dtype=np.float32)

        # action_space[车把角度，前后轮速度]
        self.actual_action_space = gymnasium.spaces.box.Box(
            low=np.array([-1.57, 0.]),
            high=np.array([1.57, 5.]),
            shape=(2,),
            dtype=np.float32)

        # Retrieve the max/min values，环境内部对action_space做了归一化
        self.action_low, self.action_high = self.actual_action_space.low, self.actual_action_space.high

        # 翻滚角, 车把角度, 后轮速度, 车与目标点距离, 车与目标点角度
        self.observation_space = gymnasium.spaces.Dict({
            "lidar": gymnasium.spaces.box.Box(low=0., high=100., shape=(180,), dtype=np.float32),
            "obs": gymnasium.spaces.box.Box(
                low=np.array([-math.pi, -1.57, -10., -100., -math.pi, -math.pi, -1.57, -10., -100., -math.pi, -math.pi, -1.57, -10., -100., -math.pi]),
                high=np.array([math.pi, 1.57, 10., 100., math.pi, math.pi, 1.57, 10., 100., math.pi, math.pi, 1.57, 10., 100., math.pi]),
                shape=(15,),
                dtype=np.float32
            ),
        })

        # 翻滚角, 翻滚角角速度, 车把角度, 车把角速度, 后轮速度, 飞轮转速, 车与目标点距离, 车与目标点角度
        self.observation_space_deprecated = gymnasium.spaces.Dict({
            "lidar": gymnasium.spaces.box.Box(low=0., high=150., shape=(360,), dtype=np.float32),
            "obs": gymnasium.spaces.box.Box(
                low=np.array([-math.pi, -50., -1.57, -50., -10., -self.max_flywheel_vel, -100., -math.pi]),
                high=np.array([math.pi, 50., 1.57, 50., 10., self.max_flywheel_vel, 100., math.pi]),
                shape=(8,),
                dtype=np.float32
            ),
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
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)

        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)  # 关闭阴影效果，透明的陀螺仪会显示出来，问题不大

        create_obstacle(self.client)
        self.grid_map = create_grid_map2(self.client, 30, 30)  # 创建网格地图

        self.smoothed_path_world = self._get_path(get_goal_pos())
        self.bicycle = BicycleLidar(self.client, self.max_flywheel_vel, obstacle_ids=[])
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        friction_coefficient = 0.5  # 摩擦系数
        # 更改地面物体的动力学参数，包括摩擦系数，-1表示所有部件
        # p.changeDynamics(plane_id, -1, lateralFriction=friction_coefficient)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.setGravity(0, 0, -10, physicsClientId=self.client)
        p.setTimeStep(1. / 120., self.client)

    def step(self, action):
        # Rescale action from [-1, 1] to original [low, high] interval
        rescaled_action = self._rescale_action(action)
        roll_angle_control = self.roll_angle_pid(self.current_roll_angle)
        self.bicycle.apply_action2(rescaled_action, -roll_angle_control)
        p.stepSimulation(physicsClientId=self.client)
        obs = self.bicycle.get_observation()

        distance_to_goal_temp = np.linalg.norm(np.array([obs[0], obs[1]]) - np.array(self.goal_temp))
        angle_to_target = my_tools.calculate_angle_to_target(obs[0], obs[1], obs[2], self.goal_temp[0], self.goal_temp[1])
        curr_obs = np.array([obs[3], obs[4], obs[5], distance_to_goal_temp, angle_to_target], dtype=np.float32)

        obs_ = np.concatenate((curr_obs, self.last_obs, self.last_last_obs))
        self.last_last_obs = self.last_obs
        self.last_obs = curr_obs

        self.current_roll_angle = obs[3]

        if self.gui:
            bike_pos, _ = p.getBasePositionAndOrientation(self.bicycle.bicycleId, physicsClientId=self.client)
            camera_distance = p.readUserDebugParameter(self.camera_distance_param)
            camera_yaw = p.readUserDebugParameter(self.camera_yaw_param)
            camera_pitch = p.readUserDebugParameter(self.camera_pitch_param)
            # p.resetDebugVisualizerCamera(cameraDistance=10, cameraYaw=0, cameraPitch=-85, cameraTargetPosition=[15, 10, 10])
            p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, bike_pos)

        # 计算奖励值
        reward = self._reward_fun(curr_obs, lidar_data=obs[6], is_collision=obs[7])
        self.prev_dist_to_goal = distance_to_goal_temp

        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            self.truncated = True

        return {"lidar": obs[6], "obs": obs_}, reward, self.terminated, self.truncated, {}

    def reset(self, seed=None, options=None):
        self.terminated = False
        self.truncated = False
        self._elapsed_steps = 0
        self.path_index = 0  # 重置路径索引
        self.goal = get_goal_pos()
        self.smoothed_path_world = self._get_path(self.goal)

        if self.smoothed_path_world is not None and len(self.smoothed_path_world) > 0:
            self.goal_temp = self.smoothed_path_world[0]  # 将目标点设置为路径的第一个点
        else:
            raise ValueError("生成的路径为空，自行车将不会移动。")
            self.goal_temp = self.bicycle_start_pos  # 如果路径为空，设置一个默认目标点，或者根据需要处理

        goal = Goal(self.client, self.goal)
        # 因为没有重置环境，每次reset后要清除先前的Goal
        if self.prev_goal_id is not None:
            p.removeBody(self.prev_goal_id, self.client)
        self.prev_goal_id = goal.id

        obs = self.bicycle.reset()
        distance_to_goal_temp = np.linalg.norm(np.array([obs[0], obs[1]]) - np.array(self.goal_temp))
        angle_to_target = my_tools.calculate_angle_to_target(obs[0], obs[1], obs[2], self.goal_temp[0], self.goal_temp[1])

        curr_obs = np.array([obs[3], obs[4], obs[5], distance_to_goal_temp, angle_to_target], dtype=np.float32)
        self.last_obs = curr_obs
        self.last_last_obs = curr_obs
        obs_ = np.concatenate((curr_obs, self.last_obs, self.last_last_obs))

        self.prev_dist_to_goal = distance_to_goal_temp
        self.current_roll_angle = obs[3]

        return {"lidar": obs[6], "obs": obs_}, {}

    def _reward_fun(self, obs, lidar_data, is_collision=False):
        self.terminated = False
        self.truncated = False
        # action [车把角度，前后轮速度]
        # obs [翻滚角, 车把角度, 后轮速度, 车与目标点距离, 车与目标点角度]
        roll_angle = obs[0]
        bicycle_vel = obs[2]
        distance_to_goal_temp = obs[3]
        # angle_to_target = obs[4]

        # ========== 平衡奖励 ==========
        roll_angle_rwd = 0.0
        if math.fabs(roll_angle) >= 0.35:
            roll_angle_rwd = -100.0
            self.terminated = True
        # ========== 平衡奖励 ==========

        # ========== 导航奖励 ==========
        diff_dist = (self.prev_dist_to_goal - distance_to_goal_temp) * 100.0
        distance_rwd = diff_dist if diff_dist > 0 else 1.5 * diff_dist
        # angle_rwd = math.cos(angle_to_target) * 0.5  # 角度对齐奖励

        goal_rwd = 0.0
        goal_temp_rwd = 0.0
        if math.fabs(distance_to_goal_temp) <= self.goal_threshold:
            goal_temp_rwd = 20.0
            self.path_index += 1  # 移动到下一个路径点
            if self.path_index < len(self.smoothed_path_world):
                self.goal_temp = self.smoothed_path_world[self.path_index]  # 更新目标为下一个路径点
            else:
                goal_rwd = 200.0  # 到达路径终点，奖励值加 200
                self.terminated = True  # 到达路径终点，可以结束 episode

        navigation_rwd = distance_rwd + goal_rwd + goal_temp_rwd
        # ========== 导航奖励 ==========

        # ========== 避障奖励 ==========
        # ========== 避障奖励 ==========

        still_penalty = 0.0
        if math.fabs(bicycle_vel) <= 0.2:
            still_penalty = -1.0
        total_reward = navigation_rwd + roll_angle_rwd + still_penalty

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

    def _get_path(self, goal_pos):
        path = a_star_pathfinding(self.grid_map, self.bicycle_start_pos, goal_pos)
        # path格式为[(row, col), (row, col), ...]
        smoothed_path_world = None  # 初始化平滑路径世界坐标
        if path:
            path_world_coords = []  # 将网格坐标路径转换为世界坐标路径，用于 Bezier 曲线平滑
            for grid_pos in path:
                world_x = grid_pos[1] * 1.0 + 1.0 / 2.0
                world_y = grid_pos[0] * 1.0 + 1.0 / 2.0
                path_world_coords.append([world_x, world_y])

            # 平滑后的路径 (世界坐标)
            smoothed_path_world = smooth_path_bezier(path_world_coords)

            # --- 添加基于距离的抽稀代码 ---
            min_dist = 3.5  # 最小距离阈值
            after_sampled_path = []  # 初始化采样后的路径
            last_point = None
            for point in smoothed_path_world:
                if last_point is None:
                    after_sampled_path.append(point)  # 第一个点总是保留
                    last_point = point
                else:
                    dist = np.linalg.norm(np.array(point) - np.array(last_point))
                    if dist > min_dist:
                        after_sampled_path.append(point)
                        last_point = point
            smoothed_path_world = after_sampled_path
            # --- 抽稀代码结束 ---

            # 可视化
            if smoothed_path_world is not None and self.gui:
                smooth_points = []
                colors = []
                for grid_pos in smoothed_path_world:  # smooth_path 已经是世界坐标
                    smooth_points.append([grid_pos[0], grid_pos[1], 0.55])  # 平滑路径稍稍抬高，避免与原始路径重叠
                    colors.append([0, 0, 1])
                p.addUserDebugPoints(smooth_points, colors, pointSize=10, physicsClientId=self.client)

        return smoothed_path_world

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
    env = gymnasium.make('BicycleMazeLidar3-v0', gui=True)
    obs, _ = env.reset()
    # check_observation_space(obs, env.observation_space)
    for i in range(10000):
        action = np.array([0.0, -1.0], np.float32)
        obs, _, terminated, truncated, infos = env.step(action)

        # if terminated or truncated:
        #     obs, _ = env.reset()
        time.sleep(1. / 120.)

    env.close()
