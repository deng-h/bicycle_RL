from utils.my_feature_extractor import MyFeatureExtractor

hyperparams = {
    "BicycleMaze-v0": dict(
        policy="MultiInputPolicy",
        normalize=dict(norm_obs=True, norm_reward=True, norm_obs_keys=['obs']),
        n_envs=10,
        n_timesteps=300000,
        learning_rate=1e-04,
        policy_kwargs=dict(
            features_extractor_class=MyFeatureExtractor,
            net_arch=dict(pi=[512, 512, 512], vf=[512, 512, 512])
        ),
        # monitor_kwargs=dict(info_keywords=('flywheel_vel',))
        env_wrapper=[{"gymnasium.wrappers.TimeLimit": {"max_episode_steps": 2500}}],
    )
}
