import gymnasium as gym
import numpy as np
import bicycle_dengh
import stable_baselines3 as sb3
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
import time
from stable_baselines3.common.monitor import Monitor
from normalize_action import NormalizeAction
from gymnasium.wrappers.normalize import NormalizeObservation
from gymnasium.wrappers.time_limit import TimeLimit
import os


def ppo(train=False):
    # 获取当前工作目录
    current_dir = os.getcwd()

    # 构建输出文件夹的路径
    models_output_dir = os.path.join(current_dir, "output", "ppo_model")
    logger_output_dir = os.path.join(current_dir, "output", "logs")

    if train:
        start_time = time.time()
        env = gym.make('BicycleDengh-v0', gui=False)
        normalized_env = NormalizeAction(env)
        normalized_env = NormalizeObservation(normalized_env)
        normalized_env = TimeLimit(normalized_env, max_episode_steps=100000)

        new_logger = configure(logger_output_dir, ["stdout", "csv", "tensorboard"])

        model = PPO(policy="MlpPolicy",
                    env=normalized_env,
                    n_steps=2048,
                    batch_size=256,
                    gae_lambda=0.98,
                    gamma=0.99,
                    n_epochs=4,
                    ent_coef=0.01,
                    # 0 for no output, 1 for info messages (such as device or wrappers used), 2 for debug messages
                    verbose=0,
                    )
        model.set_logger(new_logger)
        model.learn(total_timesteps=100000,
                    log_interval=1,)
        model.save(models_output_dir)
        # mean_reward, std_reward = evaluate_policy(model, normalized_env, n_eval_episodes=100, warn=False)
        # print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")
        del model

        end_time = time.time()
        execution_time = end_time - start_time
        execution_time_minutes = execution_time // 60
        execution_time_seconds = execution_time % 60
        print(f"训练时间为：{execution_time_minutes:.0f}分{execution_time_seconds:.0f}秒")
    else:
        env = gym.make('BicycleDengh-v0', gui=True)
        normalized_env = NormalizeAction(env)
        normalized_env = NormalizeObservation(normalized_env)
        normalized_env = TimeLimit(normalized_env, max_episode_steps=100000)

        model = PPO.load(models_output_dir)

        obs, _ = normalized_env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            # print(action)
            obs, _, terminated, truncated, _ = normalized_env.step(action)
            if terminated or truncated:
                obs, _ = normalized_env.reset()

            time.sleep(1. / 24.)


if __name__ == '__main__':
    ppo(train=False)
