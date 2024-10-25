import gymnasium as gym
import numpy as np
from gymnasium.spaces import Box
from stable_baselines3.common.env_checker import check_env
import math
from utils.normalize_action import NormalizeAction
from stable_baselines3 import PPO
from combined_extractor import CombinedExtractor
from stable_baselines3.common.evaluation import evaluate_policy


class CustomEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self):
        super(CustomEnv, self).__init__()
        self.terminated = False
        self.truncated = False

        self.action_space = Box(
            low=np.array([-1.57, 0.0, -200]),
            high=np.array([1.57, 5.0, 200]),
            shape=(3,),
            dtype=np.float32
        )

        self.observation_space = gym.spaces.Dict({
            "image": Box(low=0, high=255, shape=(36, 36, 1), dtype=np.uint8),
            "vector": Box(
                low=np.array([0.0, -math.pi, -math.pi, -15.0, -1.57, -15.0, 0.0, -200]),
                high=np.array([100.0, math.pi, math.pi, 15.0, 1.57, 15.0, 10.0, 200]),
                shape=(8,),
                dtype=np.float32
            ),
        })

    def step(self, action):
        observation = {
            "image": np.random.randint(0, 256, size=(36, 36, 1), dtype=np.uint8),
            "vector": np.array([50.0, 0.5, 0.3, 10.0, 1.0, 12.0, 5.0, 100], dtype=np.float32)
        }
        self.terminated = True
        return observation, 0, self.terminated, self.terminated, {}

    def reset(self, seed=None, options=None):
        observation = {
            "image": np.random.randint(0, 256, size=(36, 36, 1), dtype=np.uint8),
            "vector": np.array([50.0, 0.5, 0.3, 10.0, 1.0, 12.0, 5.0, 100], dtype=np.float32)
        }
        return observation, {}

    def render(self):
        pass

    def close(self):
        print("CustomEnv Closed")


if __name__ == '__main__':
    env = CustomEnv()
    env = NormalizeAction(env)
    # It will check your custom environment and output additional warnings if needed
    # check_env(env, warn=True)

    policy_kwargs = dict(
        features_extractor_class=CombinedExtractor,
    )

    model = PPO("MultiInputPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
    model.learn(100)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward:.2f}")
