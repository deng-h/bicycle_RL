import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from bicycle_dengh.resources.bicycle import Bicycle
from bicycle_dengh.resources.goal import Goal
import math
import random
from utils.my_tools import normalize_array_to_minus_one_to_one


class BicycleDenghEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, gui=False):
        self.goal = (0, 0)
        self.terminated = False
        self.truncated = False
        self.gui = gui
        self.max_flywheel_vel = 120.0
        self.prev_goal_id = None

        if gui:
            self.client = p.connect(p.GUI)
            self.camera_distance_param = p.addUserDebugParameter('camera_distance_param', 2, 60, 2)
            self.camera_yaw_param = p.addUserDebugParameter('camera_yaw_param', -180, 180, 0)
            self.camera_pitch_param = p.addUserDebugParameter('camera_pitch_param', -90, 90, -25)
            self.bicycle_vel_param = p.addUserDebugParameter('bicycle_vel_param', 0.0, 3.0, 1.0)
            self.handlebar_angle_param = p.addUserDebugParameter('handlebar_angle_param', -1.57, 1.57, 0)
            self.flywheel_param = p.addUserDebugParameter('flywheel_param', -40, 40, 0)
        else:
            self.client = p.connect(p.DIRECT)

        self.bicycle = Bicycle(client=self.client)
        self.action_space = self.bicycle.get_action_space()
        self.actual_observation_space = self.bicycle.get_observation_space()
        self.observation_space = gym.spaces.box.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            shape=(8,),
            dtype=np.float32)

        p.setTimeStep(1. / 24., self.client)
        p.setGravity(0, 0, -10, physicsClientId=self.client)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf", physicsClientId=self.client)
        # 设置飞轮速度上限
        p.changeDynamics(self.bicycle.bicycleId,
                         self.bicycle.fly_wheel_joint,
                         maxJointVelocity=self.max_flywheel_vel,
                         physicsClientId=self.client)

    def step(self, action):
        self.bicycle.apply_action(action)
        p.stepSimulation(physicsClientId=self.client)
        obs = self.bicycle.get_observation()
        current_position = (obs[0], obs[1])

        normalized_obs = normalize_array_to_minus_one_to_one(obs, self.actual_observation_space.low,
                                                             self.actual_observation_space.high)
        normalized_obs = np.array(normalized_obs, dtype=np.float32)

        if self.gui:
            bike_pos, _ = p.getBasePositionAndOrientation(self.bicycle.bicycleId, physicsClientId=self.client)
            camera_distance = p.readUserDebugParameter(self.camera_distance_param)
            camera_yaw = p.readUserDebugParameter(self.camera_yaw_param)
            camera_pitch = p.readUserDebugParameter(self.camera_pitch_param)
            p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, bike_pos)

        # 计算奖励值
        reward = self._reward_fun(obs, action)

        return normalized_obs, reward, self.terminated, self.truncated, {"current_position": current_position}

    def reset(self, seed=None, options=None):
        self.terminated = False
        self.truncated = False

        self.goal = (10, 10)
        goal = Goal(self.client, self.goal)
        # 因为没有重置环境，每次reset后要清除先前的Goal
        if self.prev_goal_id is not None:
            p.removeBody(self.prev_goal_id)
        self.prev_goal_id = goal.id

        obs = self.bicycle.reset()

        normalized_obs = normalize_array_to_minus_one_to_one(obs, self.actual_observation_space.low,
                                                             self.actual_observation_space.high)
        normalized_obs = np.array(normalized_obs, dtype=np.float32)

        return normalized_obs, {"target": self.goal}

    def _reward_fun(self, obs, action):
        self.terminated = False
        self.truncated = False
        return 0.0

    def render(self):
        pass

    def close(self):
        p.disconnect(self.client)


if __name__ == '__main__':
    env = BicycleDenghEnv(gui=False)
