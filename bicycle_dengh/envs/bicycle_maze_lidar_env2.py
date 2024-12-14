import time
import gymnasium
import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from bicycle_dengh.resources.bicycle_lidar import BicycleLidar
from bicycle_dengh.resources.goal import Goal
import math
from utils import my_tools
from simple_pid import PID
from stable_baselines3.common.env_checker import check_env
from gymnasium.wrappers import TimeLimit


"""
想法：把自行车的翻滚角控制交给PID控制器，其他控制交给RL模型
"""
class BicycleMazeLidarEnv2(gymnasium.Env):
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, gui=False):
        self.goal = (0, 0)
        self.terminated = False
        self.truncated = False
        self.gui = gui
        self.max_flywheel_vel = 120.
        self.prev_goal_id = None
        self.prev_dist_to_goal = 0.
        self.roll_angle_pid = PID(1000, 0, 0, setpoint=0.0)
        self.current_roll_angle = 0.0
        self._max_episode_steps = 15
        self._elapsed_steps = None
        self.action_space = gymnasium.spaces.box.Box(low=-1., high=1., shape=(2,), dtype=np.float32)

        # action_space[车把角度，前后轮速度, 飞轮速度]
        self.actual_action_space = gymnasium.spaces.box.Box(
            low=np.array([-1., 0.]),
            high=np.array([1., 5.]),
            shape=(2,),
            dtype=np.float32)

        # Retrieve the max/min values，环境内部对action_space做了归一化
        self.action_low, self.action_high = self.actual_action_space.low, self.actual_action_space.high

        # 翻滚角, 翻滚角角速度, 车把角度, 车把角速度, 后轮速度, 飞轮转速, 车与目标点距离, 车与目标点角度
        self.observation_space = gymnasium.spaces.Dict({
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
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)  # 关闭阴影效果，透明的陀螺仪会显示出来，问题不大

        obstacle_ids = my_tools.build_maze(self.client)
        self.bicycle = BicycleLidar(self.client, self.max_flywheel_vel, obstacle_ids)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        friction_coefficient = 0.5  # 摩擦系数
        # 更改地面物体的动力学参数，包括摩擦系数
        p.changeDynamics(plane_id, -1, lateralFriction=friction_coefficient)  # -1表示所有部件
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.setGravity(0, 0, -10, physicsClientId=self.client)
        # p.setTimeStep(1. / 24., self.client)

    def step(self, action):
        # Rescale action from [-1, 1] to original [low, high] interval
        rescaled_action = self._rescale_action(action)
        roll_angle_control = self.roll_angle_pid(self.current_roll_angle)
        self.bicycle.apply_action2(rescaled_action, -roll_angle_control)
        p.stepSimulation(physicsClientId=self.client)
        obs = self.bicycle.get_observation()

        distance_to_goal = np.linalg.norm(np.array([obs[0], obs[1]]) - np.array(self.goal))
        angle_to_target = my_tools.calculate_angle_to_target(obs[0], obs[1], obs[2], self.goal[0], self.goal[1])
        obs_ = np.array([obs[3], obs[4], obs[5], obs[6], obs[7], obs[8], distance_to_goal, angle_to_target],
                        dtype=np.float32)

        self.current_roll_angle = obs[3]

        if self.gui:
            bike_pos, _ = p.getBasePositionAndOrientation(self.bicycle.bicycleId, physicsClientId=self.client)
            camera_distance = p.readUserDebugParameter(self.camera_distance_param)
            camera_yaw = p.readUserDebugParameter(self.camera_yaw_param)
            camera_pitch = p.readUserDebugParameter(self.camera_pitch_param)
            p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, bike_pos)

        # 计算奖励值
        reward = self._reward_fun(obs_, lidar_info=obs[9], is_collision=obs[10])
        self.prev_dist_to_goal = distance_to_goal

        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            self.truncated = True

        return {"lidar": obs[9], "obs": obs_}, reward, self.terminated, self.truncated, {}

    def reset(self, seed=None, options=None):
        self.terminated = False
        self.truncated = False

        self.goal = my_tools.generate_goal()
        goal = Goal(self.client, self.goal)
        # 因为没有重置环境，每次reset后要清除先前的Goal
        if self.prev_goal_id is not None:
            p.removeBody(self.prev_goal_id, self.client)
        self.prev_goal_id = goal.id

        obs = self.bicycle.reset()
        distance_to_goal = np.linalg.norm(np.array([obs[0], obs[1]]) - np.array(self.goal))
        self.prev_dist_to_goal = distance_to_goal
        angle_to_target = my_tools.calculate_angle_to_target(obs[0], obs[1], obs[2], self.goal[0], self.goal[1])
        self.current_roll_angle = obs[3]
        obs_ = np.array([obs[3], obs[4], obs[5], obs[6], obs[7], obs[8], distance_to_goal, angle_to_target],
                        dtype=np.float32)
        self._elapsed_steps = 0

        return {"lidar": obs[9], "obs": obs_}, {}

    def _reward_fun(self, obs, lidar_info, is_collision):
        self.terminated = False
        self.truncated = False
        # action [车把角度，前后轮速度, 飞轮速度]
        # obs [翻滚角, 翻滚角角速度, 车把角度, 车把角速度, 后轮速度, 飞轮转速, 车与目标点距离, 车与目标点角度]
        roll_angle = obs[0]
        bicycle_vel = obs[4]
        distance_to_goal = obs[6]

        """平衡奖励"""
        if math.fabs(roll_angle) >= 0.35:
            self.terminated = True

        """目标点奖励"""
        goal_rwd = 0.0
        if math.fabs(distance_to_goal) <= 0.5:
            self.terminated = True
            goal_rwd = 100.0

        """静止惩罚"""
        still_penalty = 0.0
        if math.fabs(bicycle_vel) <= 0.2:
            still_penalty = -1.0

        """避障奖励"""
        # avoid_obstacle_rwd = 0.0
        # if is_collision:
        #     avoid_obstacle_rwd = -3.0
        # else:
        #     if min(lidar_info) < 0.5:
        #         avoid_obstacle_rwd = -0.5
        #     else:
        #         avoid_obstacle_rwd = 0.5

        # 距离目标点奖励
        distance_rwd = 0.0
        diff_dist_to_goal = self.prev_dist_to_goal - distance_to_goal
        if math.fabs(distance_to_goal) > 0.5:
            if math.fabs(diff_dist_to_goal) < 0.0005:
                # 没到达终点，但又不再靠近终点时
                distance_rwd = -5.0
            else:
                distance_rwd = diff_dist_to_goal * 1.5

        total_reward = distance_rwd + goal_rwd

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
    env = gymnasium.make('BicycleMazeLidar-v0', gui=True)
    obs, infos = env.reset()
    for i in range(4000):
        action = np.array([1.0, -1.0, 0.0], np.float32)
        obs, _, terminated, truncated, infos = env.step(action)
        # check_observation_space(obs, env.observation_space)

        # if terminated or truncated:
        #     obs, _ = env.reset()
        time.sleep(1)

    env.close()
