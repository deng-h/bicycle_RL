from utils.my_feature_extractor import MyFeatureExtractor
import stable_baselines3


hyperparams = {
    "BicycleMaze-v0": dict(
        policy="MultiInputPolicy",
        normalize="{'norm_obs': True, 'norm_reward': True}",
        env_wrapper=[{"rl_zoo3.wrappers.ActionSmoothingWrapper": {"smoothing_coef": 0.5}}],
        n_envs=10,
        n_timesteps=100,
        learning_rate=7.77e-05,
        policy_kwargs=dict(
            features_extractor_class=MyFeatureExtractor,
            net_arch=dict(pi=[128, 128], vf=[128, 128])
        ),
    )
}