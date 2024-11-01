import gymnasium as gym
import numpy as np
import bicycle_dengh
from stable_baselines3 import PPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
import time
from utils.normalize_action import NormalizeActionWrapper
from gymnasium.wrappers.time_limit import TimeLimit
import os
from datetime import datetime


def ppo(train=False):
    current_dir = os.getcwd()  # linux下训练前先 cd ~/denghang/bicycle-rl/BicycleDengh
    now = datetime.now()  # 获取当前时间
    formatted_time = now.strftime("%m%d_%H%M")  # 格式化时间为 mmdd_hhmm
    # env = gym.make('BicycleDengh-v0', gui=not train)
    env = gym.make('BicycleCamera-v0', gui=not train)
    # env = gym.make('BalanceBicycleDengh-v0', gui=not train)
    normalized_env = NormalizeActionWrapper(env)
    normalized_env = TimeLimit(normalized_env, max_episode_steps=1000)

    if train:
        train_model_name = "ppo_model_omni_" + formatted_time
        models_output_dir = os.path.join(current_dir, "output", "models", train_model_name)
        logger_output_dir = os.path.join(current_dir, "output", "logs")
        start_time = time.time()
        new_logger = configure(logger_output_dir, ["stdout", "csv", "tensorboard"])

        # model = PPO(policy="MlpPolicy",
        #             env=normalized_env,
        #             # 在 n_steps * n_envs 步之后更新策略
        #             n_steps=256,
        #             batch_size=256,
        #             gamma=0.99,
        #             # n_epochs 在每次策略更新中，使用相同的样本数据进行梯度下降优化的次数
        #             n_epochs=4,
        #             ent_coef=0.01,
        #             tensorboard_log=logger_output_dir,
        #             # 0 for no output, 1 for info messages (such as device or wrappers used), 2 for debug messages
        #             verbose=0,
        #             )

        checkpoint_callback = CheckpointCallback(
            save_freq=50000,
            save_path="./checkpoint/",
            name_prefix=formatted_time,
        )

        model_path = "/home/chen/denghang/bicycle-rl/BicycleDengh/output/models/ppo_model_omni_0607_1413.zip"
        model = PPO.load(path=model_path, env=normalized_env)
        model.set_logger(new_logger)
        model.learn(total_timesteps=500000,
                    log_interval=1,
                    callback=checkpoint_callback)
        model.save(models_output_dir)

        mean_reward, std_reward = evaluate_policy(model, normalized_env, n_eval_episodes=100, warn=False)
        print(f"mean_reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        del model

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"训练时间：{execution_time // 60:.0f}分{execution_time % 60:.0f}秒")
    else:
        models_dir = os.path.join(current_dir, "output", "ppo_model_omni_0607_1820")
        model = PPO.load(models_dir)
        obs, _ = normalized_env.reset()
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = normalized_env.step(action)
            if terminated or truncated:
                obs, _ = normalized_env.reset()
            time.sleep(1. / 24.)

def ppo_train():
    current_dir = os.getcwd()  # linux下训练前先 cd ~/denghang/bicycle-rl/BicycleDengh
    now = datetime.now()  # 获取当前时间
    formatted_time = now.strftime("%m%d_%H%M")  # 格式化时间为 mmdd_hhmm
    env = gym.make('BicycleDengh-v0', gui=False)
    normalized_env = NormalizeActionWrapper(env)
    normalized_env = TimeLimit(normalized_env, max_episode_steps=1000)


if __name__ == '__main__':
    ppo(train=False)