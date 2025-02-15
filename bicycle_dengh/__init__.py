# 当导入bicycle_dengh包时，该脚本会执行
# 自定义环境将被添加到 Gym 的注册表中。这样，我们就可以通过 gym.make() 的标准方法创建环境

from gymnasium.envs.registration import register

register(
    id="BicycleDengh-v0",
    entry_point='bicycle_dengh.envs:BicycleDenghEnv'
)

register(
    id="BicycleBalance-v0",
    entry_point='bicycle_dengh.envs:BicycleBalanceEnv'
)

register(
    id="BicycleCamera-v0",
    entry_point='bicycle_dengh.envs:BicycleCameraEnv'
)

register(
    id="BicycleMaze-v0",
    entry_point='bicycle_dengh.envs:BicycleMazeEnv'
)

register(
    id="BicycleMazeLidar-v0",
    entry_point='bicycle_dengh.envs:BicycleMazeLidarEnv'
)

register(
    id="BicycleMazeLidar2-v0",
    entry_point='bicycle_dengh.envs:BicycleMazeLidarEnv2'
)

register(
    id="BicycleMazeLidar3-v0",
    entry_point='bicycle_dengh.envs:BicycleMazeLidarEnv3'
)

register(
    id="BicycleMazeLidar4-v0",
    entry_point='bicycle_dengh.envs:BicycleMazeLidarEnv4'
)

register(
    id="BalanceEnvS-v0",
    entry_point='bicycle_dengh.envs:BalanceEnvS'
)
