from utils.my_feature_extractor import MyFeatureExtractorLidar, ZFeatureExtractor

hyperparams = {
    "BicycleMazeLidar2-v0": dict(
        policy="MultiInputPolicy",
        normalize=dict(norm_obs=True, norm_reward=False, norm_obs_keys=['lidar', 'obs']),
        n_envs=10,
        n_timesteps=300000,
        learning_rate=1e-4,
        batch_size=256,
        ent_coef=0.1,
        policy_kwargs=dict(
            features_extractor_class=MyFeatureExtractorLidar,
            net_arch=dict(pi=[512, 512], vf=[512, 512])
        ),
        # monitor_kwargs=dict(info_keywords=('flywheel_vel',))
    ),

    "BicycleMazeLidar3-v0": dict(
        policy="MultiInputPolicy",
        normalize=dict(norm_obs=True, norm_reward=False, norm_obs_keys=['lidar', 'obs']),
        n_envs=10,
        n_timesteps=300000,
        learning_rate=3e-4,
        batch_size=128,
        ent_coef=0.1,
        policy_kwargs=dict(
            features_extractor_class=MyFeatureExtractorLidar,
            net_arch=dict(pi=[512, 512], vf=[512, 512])
        ),
        # monitor_kwargs=dict(info_keywords=('flywheel_vel',))
    ),

    "BicycleMazeLidar5-v0": dict(
        policy="MlpPolicy",
        normalize=dict(norm_obs=True, norm_reward=True),
        n_envs=1,
        n_timesteps=300000,
        learning_rate=3e-4,
        batch_size=64,
        ent_coef=0.1,
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        ),
    ),

    "BicycleDengh-v0": dict(
        policy="MlpPolicy",
        # normalize=dict(norm_obs=True, norm_reward=True),
        n_envs=10,
        n_timesteps=500000,
        learning_rate=1e-4,
        batch_size=128,
        # ent_coef=0.01,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        # monitor_kwargs=dict(info_keywords=('flywheel_vel',))
    ),

    "ZBicycleBalanceEnv-v0": dict(
        policy="MlpPolicy",
        normalize=dict(norm_obs=True, norm_reward=False),
        n_envs=6,
        n_steps=256,
        batch_size=1536,  # n_steps * n_envs
        gamma=0.99,
        n_epochs=4,
        ent_coef=0.01,
        n_timesteps=1000000,
        learning_rate=3e-4,
        policy_kwargs=dict(
            net_arch=dict(pi=[64, 64], vf=[64, 64])
        ),
    ),

    "ZBicycleNaviEnv-v0": dict(
        policy="MultiInputPolicy",
        normalize=dict(norm_obs=True, norm_reward=False, norm_obs_keys=['lidar', 'bicycle']),
        n_envs=6,
        n_steps=256,
        batch_size=1536,  # n_steps * n_envs
        gamma=0.99,
        n_epochs=4,
        ent_coef=0.01,
        n_timesteps=1000000,
        learning_rate=3e-4,
        policy_kwargs=dict(
            features_extractor_class=ZFeatureExtractor,
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
    ),

    "BicycleDmzEnv-v0": dict(
        policy="MultiInputPolicy",
        normalize=dict(norm_obs=True, norm_reward=False, norm_obs_keys=['lidar', 'bicycle']),
        n_envs=12,
        n_steps=2048,
        batch_size=24576,  # n_steps * n_envs
        ent_coef=0.02,
        n_timesteps=600000,
        policy_kwargs=dict(
            features_extractor_class=ZFeatureExtractor,
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
    ),
}
