from utils.my_feature_extractor import MyFeatureExtractorLidar

hyperparams = {
    "BicycleMazeLidar2-v0": dict(
        policy="MultiInputPolicy",
        normalize=dict(norm_obs=True, norm_reward=False, norm_obs_keys=['lidar', 'obs']),
        n_envs=32,
        n_timesteps=300000,
        learning_rate=1e-4,
        batch_size=512,
        policy_kwargs=dict(
            features_extractor_class=MyFeatureExtractorLidar,
            net_arch=dict(pi=[512, 512], vf=[512, 512])
        ),
        # monitor_kwargs=dict(info_keywords=('flywheel_vel',))
    )
}