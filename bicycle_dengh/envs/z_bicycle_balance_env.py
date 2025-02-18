import gymnasium
import numpy as np
import pybullet as p
import pybullet_data
from bicycle_dengh.resources.z_bicycle import ZBicycle
import math


class ZBicycleBalanceEnv(gymnasium.Env):
    metadata = {'render_modes': ['rgb_array']}

    def __init__(self, gui=False):
        self.terminated = False
        self.truncated = False
        self.gui = gui
        self.max_flywheel_vel = 120.
        self._max_episode_steps = 5000
        self._elapsed_steps = None

        self.action_space = gymnasium.spaces.box.Box(low=-1., high=1., shape=(1,), dtype=np.float32)

        # [飞轮]
        self.actual_action_space = gymnasium.spaces.box.Box(
            low=np.array([-self.max_flywheel_vel]),
            high=np.array([self.max_flywheel_vel]),
            shape=(1,),
            dtype=np.float32
        )

        self.action_low, self.action_high = self.actual_action_space.low, self.actual_action_space.high

        # [翻滚角, 翻滚角角速度, 车把角度, 车把角速度, 后轮速度, 飞轮速度]
        self.observation_space = gymnasium.spaces.box.Box(
            low=np.array([-math.pi, -15.0, -1.57, -15.0, 0.0, -self.max_flywheel_vel]),
            high=np.array([math.pi, 15.0, 1.57, 15.0, 10.0, self.max_flywheel_vel]),
            shape=(6,),
            dtype=np.float32
        )

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

        self.bicycle = ZBicycle(self.client, self.max_flywheel_vel)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        friction_coefficient = 1.5
        # 更改地面物体的动力学参数，包括摩擦系数
        p.changeDynamics(plane_id, -1, lateralFriction=friction_coefficient)  # -1表示所有部件
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        p.setGravity(0, 0, -10, physicsClientId=self.client)
        p.setTimeStep(1. / 24., self.client)

    def step(self, action):
        rescaled_action = self._rescale_action(action)
        input_action = [0, 0, rescaled_action[0]]
        self.bicycle.apply_action(input_action)
        p.stepSimulation(physicsClientId=self.client)

        obs = self.bicycle.get_observation()
        ret_obs = [obs[3], obs[4], obs[5], obs[6], obs[7], obs[8]]
        reward = self._reward_fun(ret_obs)

        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            self.truncated = True

        if self.gui:
            bike_pos, _ = p.getBasePositionAndOrientation(self.bicycle.bicycleId, physicsClientId=self.client)
            camera_distance = p.readUserDebugParameter(self.camera_distance_param)
            camera_yaw = p.readUserDebugParameter(self.camera_yaw_param)
            camera_pitch = p.readUserDebugParameter(self.camera_pitch_param)
            p.resetDebugVisualizerCamera(camera_distance, camera_yaw, camera_pitch, bike_pos)

        return np.array(ret_obs, dtype=np.float32), reward, self.terminated, self.truncated, {}

    def reset(self, seed=None, options=None):
        self.terminated = False
        self.truncated = False
        self._elapsed_steps = 0
        obs = self.bicycle.reset()
        ret_obs = [obs[3], obs[4], obs[5], obs[6], obs[7], obs[8]]
        return np.array(ret_obs, dtype=np.float32), {}

    def _reward_fun(self, obs):
        self.terminated = False
        self.truncated = False

        roll_angle = obs[0]

        """平衡奖励"""
        balance_rwd = 0.0
        if math.fabs(roll_angle) >= 0.35:
            self.terminated = True
            balance_rwd = -10.0
        else:
            # 计算奖励值，倾角越小，奖励越大
            balance_rwd = 1.0 - (math.fabs(roll_angle) / 0.35) * 2.0
            # 限制奖励值在范围[-max_reward, max_reward]之间
            balance_rwd = max(-1.0, min(1.0, balance_rwd))

        # r2 = -0.001 * math.fabs(wheel_vel)  # 惯性轮速度惩罚

        balance_rwd_dui_bi = 0.0
        # if math.fabs(roll_angle) >= 0.35:
        #     self.terminated = True
        #     balance_rwd_dui_bi = -10.0
        # elif math.fabs(roll_angle) >= 0.25:
        #     balance_rwd_dui_bi = -1.0
        # elif math.fabs(roll_angle) >= 0.15:
        #     balance_rwd_dui_bi = -0.5
        # else:
        #     balance_rwd_dui_bi = 1.0
        # return balance_rwd + r2

        return balance_rwd

    def render(self):
        pass

    def _rescale_action(self, scaled_action):
        """
        Rescale the action from [-1, 1] to [low, high] (no need for symmetric action space)
        :param scaled_action: (np.ndarray)
        :return: (np.ndarray)
        """
        return self.action_low + (0.5 * (scaled_action + 1.0) * (self.action_high - self.action_low))


if __name__ == '__main__':
    env = gymnasium.make('BicycleBalance-v0', gui=True)
    obs, _ = env.reset()
    for i in range(40000):
        action = np.array([0.5], np.float32)
        obs, _, terminated, truncated, _ = env.step(action)

        if terminated or truncated:
            obs, _ = env.reset()
        # time.sleep(1)

    env.close()
