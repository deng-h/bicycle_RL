from datetime import datetime
import numpy as np
import gymnasium as gym
import bicycle_dengh
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, CallbackList
from utils.my_feature_extractor import MyFeatureExtractor
import os
import time


# 多线程训练
def vec_env_train_in_linux():
    current_dir = os.getcwd()  # linux下训练前先 cd ~/denghang/bicycle-rl/BicycleDengh
    models_path = os.path.join(current_dir, "output", "models")
    logger_path = os.path.join(current_dir, "output", "logs")
    checkpoints_path = os.path.join(current_dir, "output", "checkpoints")

    formatted_time = datetime.now().strftime("%m%d%H%M")  # 格式化时间为 mmddhhmm
    start_time = time.time()
    model_name = "ppo_maze_" + formatted_time

    env = make_vec_env("BicycleMaze-v0", n_envs=4, vec_env_cls=SubprocVecEnv)
    env = VecNormalize(env, norm_obs=True, norm_reward=True)

    eval_env = DummyVecEnv([lambda: gym.make("BicycleMaze-v0", gui=False)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=True)
    eval_callback = EvalCallback(eval_env=eval_env,
                                 eval_freq=max(10000 // 4, 1),  # 每10000步评估一下
                                 best_model_save_path=models_path,
                                 log_path=logger_path,
                                 deterministic=True,
                                 render=False)

    checkpoint_callback = CheckpointCallback(
                                save_freq=max(100000 // 4, 1),  # 每100000步保存一下
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

    model = PPO(policy="MultiInputPolicy",
                env=env,
                learning_rate=0.0001,
                verbose=1,
                policy_kwargs=policy_kwargs,
                tensorboard_log=logger_path,
                )

    # model_path = "/home/chen/denghang/bicycle-rl/BicycleDengh/output/models/ppo_multiprocess_maze_1030_1054.zip"
    # model = PPO.load(path=model_path, env=env)
    # print(f"网络的架构:{model.policy}")

    model.learn(total_timesteps=100000,
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
    vec_env_train_in_linux()
