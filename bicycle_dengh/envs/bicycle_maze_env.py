import os
import time
import gymnasium
import numpy as np
import pybullet as p
import pybullet_data
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from bicycle_dengh.resources.bicycle_camera import BicycleCamera
from bicycle_dengh.resources.goal import Goal
import math
from utils import my_tools
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor


def calculate_roll_angle_rwd(roll_angle):
    if math.fabs(roll_angle) >= 0.26:
        return True, -5.0

    # 计算奖励值，倾角越小，奖励越大
    # 奖励值范围从0.5到-0.5，倾角越小奖励值越接近0.5
    reward = 0.5 - (math.fabs(roll_angle) / 0.26) * 1.0  # 将15度映射到奖励范围[0.5, -0.5]

    # 限制奖励值在范围[-0.5, 0.5]之间
    reward = max(-0.5, min(0.5, reward))

    return False, reward


class BicycleMazeEnv(gymnasium.Env):
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, gui=False):
        self.goal = (0, 0)
        self.terminated = False
        self.truncated = False
        self.gui = gui
        self.max_flywheel_vel = 120.0
        self.prev_goal_id = None
        self.collision_times = 0
        self.prev_dist_to_goal = 0.0
        self.action_space = gymnasium.spaces.box.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        # action_space[车把角度，前后轮速度, 飞轮速度]
        self.actual_action_space = gymnasium.spaces.box.Box(
            low=np.array([-1.57, 0.0, -self.max_flywheel_vel]),
            high=np.array([1.57, 5.0, self.max_flywheel_vel]),
            shape=(3,),
            dtype=np.float32)

        # Retrieve the max/min values，环境内部对action_space做了归一化
        self.action_low, self.action_high = self.actual_action_space.low, self.actual_action_space.high

        # 机器人与目标点距离, 机器人与目标点的角度, 翻滚角, 翻滚角角速度, 车把角度, 车把角速度, 后轮速度, 飞轮速度
        self.observation_space = gymnasium.spaces.Dict({
            "image": gymnasium.spaces.box.Box(low=0, high=255, shape=(3, 128, 128), dtype=np.float32),
            "obs": gymnasium.spaces.box.Box(
                low=np.array([0.0, -math.pi, -math.pi, -15.0, -1.57, -15.0, -10.0, -self.max_flywheel_vel]),
                high=np.array([100.0, math.pi, math.pi, 15.0, 1.57, 15.0, 10.0, self.max_flywheel_vel]),
                shape=(8,),
                dtype=np.float32
            ),
            "last_action": gymnasium.spaces.box.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
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
        self.bicycle = BicycleCamera(self.client, self.max_flywheel_vel, obstacle_ids)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.setGravity(0, 0, -10, physicsClientId=self.client)
        p.setTimeStep(1. / 24., self.client)

    def step(self, action):
        # Rescale action from [-1, 1] to original [low, high] interval
        rescaled_action = self._rescale_action(action)
        self.bicycle.apply_action(rescaled_action)
        p.stepSimulation(physicsClientId=self.client)
        obs = self.bicycle.get_observation()

        distance_to_goal = np.linalg.norm(np.array([obs[0], obs[1]]) - np.array(self.goal))
        angle_to_target = my_tools.calculate_angle_to_target(obs[0], obs[1], obs[2], self.goal[0], self.goal[1])
        image_obs = obs[9]
        # 机器人位置与目标位置距离, 机器人位置与目标位置夹角, 翻滚角, 翻滚角角速度, 车把角度, 车把角速度, 后轮速度, 飞轮速度
        vector_obs = np.array([distance_to_goal, angle_to_target, obs[3], obs[4], obs[5], obs[6], obs[7], obs[8]],
                              np.float32)

        if self.gui:
            bike_pos, _ = p.getBasePositionAndOrientation(self.bicycle.bicycleId, physicsClientId=self.client)
            camera_distance = p.readUserDebugParameter(self.camera_distance_param)
            camera_yaw = p.readUserDebugParameter(self.camera_yaw_param)
            camera_pitch = p.readUserDebugParameter(self.camera_pitch_param)
            p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, bike_pos)

        # 计算奖励值
        reward = self._reward_fun(vector_obs, is_collision=obs[10])
        self.prev_dist_to_goal = distance_to_goal

        return ({"image": image_obs, "obs": vector_obs, "last_action": action},
                reward, self.terminated, self.truncated, {})

    def reset(self, seed=None, options=None):
        self.terminated = False
        self.truncated = False
        self.collision_times = 0

        self.goal = my_tools.generate_goal()
        goal = Goal(self.client, self.goal)
        # 因为没有重置环境，每次reset后要清除先前的Goal
        if self.prev_goal_id is not None:
            p.removeBody(self.prev_goal_id, self.client)
        self.prev_goal_id = goal.id

        # 机器人位置与目标位置差x, 机器人位置与目标位置差y, 偏航角, 翻滚角, 翻滚角角速度, 车把角度, 车把角速度, 后轮速度, 飞轮速度, 深度图
        obs = self.bicycle.reset()
        distance_to_goal = np.linalg.norm(np.array([obs[0], obs[1]]) - np.array(self.goal))
        self.prev_dist_to_goal = distance_to_goal
        angle_to_target = my_tools.calculate_angle_to_target(obs[0], obs[1], obs[2], self.goal[0], self.goal[1])

        image_obs = obs[9]
        vector_obs = np.array([distance_to_goal, angle_to_target, obs[3], obs[4], obs[5], obs[6], obs[7], obs[8]],
                              np.float32)
        last_action = np.zeros(3, np.float32)

        return {"image": image_obs, "obs": vector_obs, "last_action": last_action}, {}

    def _reward_fun(self, obs, is_collision):
        self.terminated = False
        self.truncated = False

        # action [车把角度，前后轮速度, 飞轮速度]
        # obs [机器人与目标点距离, 机器人与目标点的角度, 翻滚角, 翻滚角角速度, 车把角度, 车把角速度, 后轮速度, 飞轮速度]
        roll_angle = obs[2]
        bicycle_vel = obs[6]

        self.terminated, balance_rwd = calculate_roll_angle_rwd(roll_angle)

        #  到达目标点奖励
        goal_rwd = 0.0
        if math.fabs(obs[0]) <= 0.5:
            self.terminated = True
            goal_rwd = 100.0

        # 静止惩罚
        still_penalty = 0.0
        if math.fabs(bicycle_vel) <= 0.2:
            still_penalty = -1.0

        # 距离目标点奖励
        distance_rwd = 0.0
        diff_dist_to_goal = self.prev_dist_to_goal - obs[0]
        if math.fabs(obs[0]) > 0.5:
            if math.fabs(diff_dist_to_goal) < 0.0005:
                # 没到达终点，但又不再靠近终点时
                distance_rwd = -5.0
            else:
                distance_rwd = diff_dist_to_goal * 1.2

        collision_penalty = 0.0
        if is_collision:
            self.collision_times += 1
            collision_penalty = -2.0
            if self.collision_times > 60:
                self.terminated = True
                collision_penalty = -100.0

        total_reward = goal_rwd + distance_rwd + balance_rwd + still_penalty + collision_penalty

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


# 非向量化的环境用于验证自定义环境的正确性
def no_vec_env():
    env = gymnasium.make('BicycleMaze-v0')
    # 环境内部对action_space做了归一化，所以这里不需要再做归一化了
    # min_action = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
    # max_action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    # env = RescaleAction(env, min_action=min_action, max_action=max_action)
    check_env(env, warn=True)


# 向量化的环境用于训练
def vec_env():
    env = make_vec_env("BicycleMaze-v0", n_envs=4)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)


if __name__ == '__main__':
    no_vec_env()
