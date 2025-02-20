from utils.my_feature_extractor import MyFeatureExtractorLidar, ZFeatureExtractor

hyperparams = {
    "ZBicycleNaviEnv-v0": dict(
        policy="MlpPolicy",
        normalize=dict(norm_obs=True, norm_reward=False),
        n_envs=20,
        n_steps=3000,
        batch_size=60000,  # n_steps * n_envs
        gamma=0.99,
        n_epochs=4,
        n_timesteps=500000,
        policy_kwargs=dict(
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
    ),

    "BicycleDmzEnv-v0": dict(
        policy="MultiInputPolicy",
        normalize=dict(norm_obs=True, norm_reward=False, norm_obs_keys=['lidar', 'bicycle']),
        n_envs=10,
        n_steps=1024,
        batch_size=25600,  # n_steps * n_envs
        ent_coef=0.01,
        n_epochs=4,
        n_timesteps=500000,
        policy_kwargs=dict(
            features_extractor_class=ZFeatureExtractor,
            net_arch=dict(pi=[256, 256], vf=[256, 256])
        ),
        # monitor_kwargs=dict(info_keywords=('reward',))
    ),

}