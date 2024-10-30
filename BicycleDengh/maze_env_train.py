from datetime import datetime
import numpy as np
import gymnasium as gym
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import bicycle_dengh
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from typing import Callable, List
import os
import time


class MyFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # 输入给net_arch网络的特征维度定为256=248+8，248是图像的特征，8是自行车的状态向量
        super().__init__(observation_space, features_dim=256)
        self.image_model = nn.Sequential(
            # height_in=128, width_in=128, CHW
            # 在线卷积池化公式计算器 http://www.sqflash.com/cal.html
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 16x64x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x32x32
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 32x16x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x8x8
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64x4x4
            nn.ReLU(),
            nn.Flatten(),  # 展平为64 * 4 * 4维
            nn.Linear(64 * 4 * 4, 248)  # 根据卷积后的特征图大小计算输入特征维度
        )
        # 定义状态特征提取器
        self.state_model = nn.Sequential(
            nn.Linear(8, 16),  # 假设输入状态为8维
            nn.ReLU(),
            nn.Linear(16, 8),  # 提取出8维的状态特征
            nn.ReLU()
        )

    def forward(self, observations) -> th.Tensor:
        image_obs = observations["image"]  # 从输入中提取图像和状态特征
        vector_obs = observations["vector"]
        image_output = self.image_model(image_obs)  # 通过图像特征提取器处理图像特征
        state_output = self.state_model(vector_obs)  # 通过状态特征提取器处理状态特征
        combined_features = th.cat([image_output, state_output], dim=1)  # 拼接图像特征和状态特征
        return combined_features


def make_env(env_id: str, rank: int, seed: int = 0) -> Callable:
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID 环境的ID，也就是环境的名称，如CartPole-v1
    :param seed: (int) the initial seed for RNG
    :param rank: (int) index of the subprocess
    :return: (Callable)
    """

    def _init() -> gym.Env:
        env = gym.make(env_id, gui=False)
        # 环境内部对action_space做了归一化，所以这里不需要再做归一化了
        # min_action = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
        # max_action = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        # env = RescaleAction(env, min_action=min_action, max_action=max_action)
        env.reset(seed=seed + rank)
        return env

    set_random_seed(seed)
    return _init


# 多线程训练
def vec_env_train():
    current_dir = os.getcwd()  # linux下训练前先 cd ~/denghang/bicycle-rl/BicycleDengh

    formatted_time = datetime.now().strftime("%m%d_%H%M")  # 格式化时间为 mmdd_hhmm
    start_time = time.time()
    model_name = "ppo_multiprocess_maze_" + formatted_time

    models_path = os.path.join(current_dir, "output", "models")
    logger_path = os.path.join(current_dir, "output", "logs")
    checkpoints_path = os.path.join(current_dir, "output", "checkpoints")

    env = make_vec_env("BicycleMaze-v0", n_envs=4, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    eval_env = DummyVecEnv([lambda: gym.make("BicycleMaze-v0", gui=False)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
    eval_callback = EvalCallback(eval_env=eval_env,
                                 best_model_save_path=models_path,
                                 log_path=logger_path,
                                 eval_freq=5000,
                                 deterministic=True,
                                 render=False)

    checkpoint_callback = CheckpointCallback(
                                save_freq=max(100000 // 4, 1),
                                save_path=checkpoints_path,
                                name_prefix=formatted_time,
                                save_vecnormalize=True,
                                verbose=1,
                                )

    callback = CallbackList([checkpoint_callback, eval_callback])

    policy_kwargs = dict(
        features_extractor_class=MyFeatureExtractor,
        net_arch=dict(pi=[128, 128], vf=[128, 128]),
    )

    # model = PPO(policy="MultiInputPolicy",
    #             env=env,
    #             learning_rate=0.0001,
    #             verbose=1,
    #             policy_kwargs=policy_kwargs,
    #             tensorboard_log=logger_path,
    #             )

    model_path = "/home/chen/denghang/bicycle-rl/BicycleDengh/output/models/ppo_multiprocess_maze_1030_1054.zip"
    model = PPO.load(path=model_path, env=env)
    # print(f"网络的架构:{model.policy}")

    model.learn(total_timesteps=10000,
                callback=callback,
                progress_bar=True,
                tb_log_name="PPO_" + formatted_time,
                )

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")

    model_save_path = os.path.join(models_path, model_name)
    model.save(model_save_path)
    del model

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"训练时间：{execution_time // 60:.0f}分{execution_time % 60:.0f}秒")


def play():
    env = gym.make("BicycleMaze-v0", gui=True)
    current_dir = os.getcwd()  # linux下训练前先 cd ~/denghang/bicycle-rl/BicycleDengh
    model_path = os.path.join(current_dir, "output", "models", "ppo_multiprocess_maze_1029_1456")
    model = PPO.load(model_path)

    obs, _ = env.reset()
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            obs, _ = env.reset()
            pass
        time.sleep(1. / 24.)


if __name__ == '__main__':
    vec_env_train()
