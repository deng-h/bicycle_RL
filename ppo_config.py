from utils.my_feature_extractor import MyFeatureExtractorLidar, ZFeatureExtractor

hyperparams = {
    "ZBicycleNaviEnv-v0": dict(
        policy="MlpPolicy",
        normalize=dict(norm_obs=True, norm_reward=True),
        n_envs=6,
        n_steps=3000,
        batch_size=18000,  # n_steps * n_envs
        gamma=0.99,
        n_epochs=4,
        ent_coef=0.05,
        n_timesteps=1000000,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
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
        n_steps=128,
        batch_size=768,  # n_steps * n_envs
        gamma=0.99,
        n_epochs=4,
        n_timesteps=500000,
        policy_kwargs=dict(
            # net_arch=dict(pi=[64, 64], vf=[64, 64])
        ),
    ),

    "BicycleDmzEnv-v0": dict(
        policy="MultiInputPolicy",
        normalize=dict(norm_obs=True, norm_reward=False, norm_obs_keys=['lidar', 'bicycle']),
        n_envs=10,
        n_steps=1024,
        batch_size=10240,  # n_steps * n_envs
        ent_coef=0.01,
        n_epochs=4,
        n_timesteps=500000,
        policy_kwargs=dict(
            features_extractor_class=ZFeatureExtractor,
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        # monitor_kwargs=dict(info_keywords=('reward',))
    ),

    "BicycleFinalEnv-v0": dict(
        policy="MlpPolicy",
        normalize=dict(norm_obs=True, norm_reward=True),
        n_envs=6,
        n_steps=5000,
        batch_size=30000,  # n_steps * n_envs
        gamma=0.99,
        n_epochs=5,
        ent_coef=0.02,
        learning_rate=3e-4,
        n_timesteps=5000000,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
    ),
}
