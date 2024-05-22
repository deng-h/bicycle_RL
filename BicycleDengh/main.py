import gymnasium as gym
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
import bicycle_dengh
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.logger import configure
import time
from normalize_action import NormalizeAction
from gymnasium.wrappers.time_limit import TimeLimit
import os
import winsound
import csv


def ppo(train=False):
    current_dir = os.getcwd()
    models_output_dir = os.path.join(current_dir, "output", "ppo_model")
    logger_output_dir = os.path.join(current_dir, "output", "logs")

    # env = gym.make('BicycleDengh-v0', gui=not train)
    env = gym.make('BalanceBicycleDengh-v0', gui=not train)
    normalized_env = NormalizeAction(env)
    normalized_env = TimeLimit(normalized_env, max_episode_steps=1000)

    if train:
        start_time = time.time()
        new_logger = configure(logger_output_dir, ["stdout", "csv", "tensorboard"])

        model = PPO(policy="MlpPolicy",
                    env=normalized_env,
                    # 在 n_steps * n_envs 步之后更新策略
                    n_steps=512,
                    batch_size=512,
                    gamma=0.99,
                    # n_epochs 在每次策略更新中，使用相同的样本数据进行梯度下降优化的次数
                    n_epochs=4,
                    ent_coef=0.01,
                    tensorboard_log=logger_output_dir,
                    # 0 for no output, 1 for info messages (such as device or wrappers used), 2 for debug messages
                    verbose=0,
                    )
        model.set_logger(new_logger)
        model.learn(total_timesteps=400000,
                    log_interval=1,)
        model.save(models_output_dir)
        # mean_reward, std_reward = evaluate_policy(model, normalized_env, n_eval_episodes=100, warn=False)
        # print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        del model
        for _ in range(3):
            winsound.Beep(300, 500)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"训练时间：{execution_time // 60:.0f}分{execution_time % 60:.0f}秒")
    else:
        model = PPO.load(models_output_dir)
        obs, _ = normalized_env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = normalized_env.step(action)
            if terminated or truncated:
                obs, _ = normalized_env.reset()

            time.sleep(1. / 24.)


if __name__ == '__main__':
    ppo(train=False)
