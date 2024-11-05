import sys
from rl_zoo3.train import train


if __name__ == "__main__":
    # sys.argv = ["python",
    #             "--algo", "ppo",
    #             "--env", "BicycleMaze-v0",
    #             "--conf-file", "ppo_config",
    #             "--tensorboard-log", "./logs/tensorboard/",
    #             "--vec-env", "subproc",
    #             "--progress"]

    train()
