from datetime import datetime
import numpy as np
import gymnasium as gym
import torch as th
from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import bicycle_dengh
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Callable
import os
import time


# 这个类的主要功能是接收多种类型的观测数据，针对每种类型设计不同的特征提取方式，并将提取后的特征拼接成一个统一的Tensor，供强化学习模型输入
# 这种设计适用于复杂的观测空间，例如同时包含图像和向量信息的环境
class CombinedExtractor(BaseFeaturesExtractor):
    # 这里传入的 observation_space 是一个 gym.spaces.Dict，代表了不同类型的观测空间。每个键（key）对应一种观测类型，如图像或向量
    def __init__(self, observation_space: gym.spaces.Dict):
        # 先初始化父类 BaseFeaturesExtractor，并且暂时设置 features_dim=1，因为我们还不知道最终的特征维度，需要通过遍历观测空间来确定
        super().__init__(observation_space, features_dim=1)
        extractors = {}
        total_concat_size = 0
        # 我们需要知道该提取器输出的大小，因此，遍历所有空间并计算输出特征大小并构建特征提取器
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                extractors[key] = nn.Sequential(
                    # height_in = 120, width_in = 160, CHW
                    # width_out = (width_in - kernel_size + 2 * padding) / stride + 1 （向下取整）
                    # height_out = (height_in - kernel_size + 2 * padding) / stride + 1 （向下取整）
                    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # 16x60x80
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),  # 16x30x40
                    nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 32x15x20
                    nn.ReLU(),
                    nn.MaxPool2d(kernel_size=2),  # 32x7x10
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64x4x5
                    nn.ReLU(),
                    nn.Flatten(),  # 展平为64 * 4 * 5维，即1280维
                )
                total_concat_size += 64 * 4 * 5
            elif key == "vector":
                # 对于向量类型的观测空间，使用一个全连接层 nn.Linear() 将输入向量映射到8维的特征空间，并更新 total_concat_size
                extractors[key] = nn.Linear(subspace.shape[0], 8)
                total_concat_size += 8

        # 特征提取器存储
        self.extractors = nn.ModuleDict(extractors)

        # 手动更新最终的特征维度，_features_dim是父类的成员变量
        self._features_dim = total_concat_size

    # 前向传播函数用于将输入的观测数据通过各个特征提取器，生成拼接后的特征向量，给后面的net_arch网络
    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []
        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            encoded_tensor = extractor(observations[key])
            # 对于每个观测类型，调用对应的提取器对其进行处理，并将结果添加到 encoded_tensor_list 列表中
            # print(f"Output shape for {key}: {encoded_tensor.shape}")
            encoded_tensor_list.append(encoded_tensor)
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        # 将所有提取到的特征张量在第1维（特征维度）进行拼接，形成一个统一的特征张量，供后续的强化学习算法使用
        return th.cat(encoded_tensor_list, dim=1)


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


if __name__ == '__main__':
    num_envs = 4  # 定义并行环境数量
    current_dir = os.getcwd()  # linux下训练前先 cd ~/denghang/bicycle-rl/BicycleDengh
    train = True

    if train:
        # 多线程训练
        env = SubprocVecEnv([make_env("BicycleMaze-v0", i) for i in range(num_envs)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        # 单线程训练
        # env = gym.make("BicycleMaze-v0", gui=False)

        policy_kwargs = dict(
            features_extractor_class=CombinedExtractor,
            net_arch=dict(pi=[128, 128], vf=[128, 128]),
        )

        formatted_time = datetime.now().strftime("%m%d_%H%M")  # 格式化时间为 mmdd_hhmm
        model_name = "ppo_multiprocess_maze_" + formatted_time
        models_output_dir = os.path.join(current_dir, "output", "models", model_name)
        logger_output_dir = os.path.join(current_dir, "output", "logs", model_name)

        start_time = time.time()
        model = PPO(policy="MultiInputPolicy",
                    env=env,
                    policy_kwargs=policy_kwargs,
                    verbose=1,
                    tensorboard_log=logger_output_dir,
                    )

        model.learn(total_timesteps=100000)

        mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")

        model.save(models_output_dir)
        del model

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"训练时间：{execution_time // 60:.0f}分{execution_time % 60:.0f}秒")
    else:
        env = gym.make("BicycleMaze-v0", gui=True)

        model_path = os.path.join(current_dir, "output", "models", "ppo_multiprocess_maze_1028_1404")
        model = PPO.load(model_path)

        obs, _ = env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
            time.sleep(1. / 24.)

