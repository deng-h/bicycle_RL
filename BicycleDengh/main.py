import gymnasium as gym
from stable_baselines3 import PPO
import time
from utils.normalize_action import NormalizeAction
from gymnasium.wrappers.time_limit import TimeLimit
import os
from datetime import datetime


def ppo(train=False):
    # linux下训练前先到这个路径下 cd ~/denghang/bicycle-rl/BicycleDengh
    current_dir = os.getcwd()
    now = datetime.now()
    formatted_time = now.strftime("%m%d_%H%M")  # 格式化时间为 mmdd_hhmm
    train_model_name = "ppo_model_omni_" + formatted_time

    env = gym.make('BicycleDengh-v0', gui=not train)
    # env = gym.make('BalanceBicycleDengh-v0', gui=not train)  # 单纯平衡版

    normalized_env = NormalizeAction(env)
    normalized_env = TimeLimit(normalized_env, max_episode_steps=1000)

    if train:
        models_output_dir = os.path.join(current_dir, "output", train_model_name)
        logger_output_dir = os.path.join(current_dir, "output", "logs")

        start_time = time.time()



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


if __name__ == '__main__':
    ppo(train=False)

