from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch as th


# 自定义的特征提取器
class MyFeatureExtractor1(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # 输入给net_arch网络的特征维度=图像特征维度+自行车的状态向量维度
        super().__init__(observation_space, features_dim=256+8)
        self.image_model = nn.Sequential(
            # height_in=128, width_in=128, CHW, 在线卷积池化公式计算器 http://www.sqflash.com/cal.html
            nn.Conv2d(5, 16, kernel_size=3, stride=2, padding=1),  # 16x64x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x32x32
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 32x16x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x8x8
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64x4x4
            nn.ReLU(),
            nn.Flatten(),  # 展平为64 * 4 * 4维
            nn.Linear(64 * 4 * 4, 256)  # 根据卷积后的特征图大小计算输入特征维度
        )

        # 定义状态特征提取器
        self.state_model = nn.Sequential(
            nn.Linear(8, 64),
        )

    def forward(self, observations) -> th.Tensor:
        image_obs = observations["image"]  # 从输入中提取图像和状态特征
        obs = observations["obs"]
        # last_action = observations["last_action"]

        image_output = self.image_model(image_obs)  # 通过图像特征提取器处理图像特征
        # state_output = self.state_model(obs)  # 通过状态特征提取器处理状态特征
        combined_features = th.cat([image_output, obs], dim=1)  # 拼接图像特征和状态特征
        return combined_features


class MyFeatureExtractor2(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # 输入给net_arch网络的特征维度=图像特征维度+自行车的状态向量维度
        super().__init__(observation_space, features_dim=256+64)
        self.image_model = nn.Sequential(
            # height_in=128, width_in=128, CHW, 在线卷积池化公式计算器 http://www.sqflash.com/cal.html
            nn.Conv2d(5, 16, kernel_size=3, stride=2, padding=1),  # 16x64x64
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x32x32
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # 32x16x16
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x8x8
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # 64x4x4
            nn.ReLU(),
            nn.Flatten(),  # 展平为64 * 4 * 4维
            nn.Linear(64 * 4 * 4, 256)  # 根据卷积后的特征图大小计算输入特征维度
        )
        # 定义状态特征提取器
        self.state_model = nn.Sequential(
            nn.Linear(8, 64),
            nn.LeakyReLU()
        )

        # Attention mechanism: Learnable weights for image and state features
        self.image_attention = nn.Linear(256, 1)  # For image output
        self.state_attention = nn.Linear(64, 1)   # For state output

        # Final fully connected layer for the combined features
        self.fc_combined = nn.Linear(256 + 64, 320)

    def forward(self, observations) -> th.Tensor:
        image_obs = observations["image"]  # 从输入中提取图像和状态特征
        obs = observations["obs"]

        image_output = self.image_model(image_obs)  # 通过图像特征提取器处理图像特征
        state_output = self.state_model(obs)  # 通过状态特征提取器处理状态特征

        # Compute attention scores for image and state
        image_att = th.sigmoid(self.image_attention(image_output))  # Shape: [batch_size, 1]
        state_att = th.sigmoid(self.state_attention(state_output))  # Shape: [batch_size, 1]

        # Normalize the attention weights to sum to 1
        att_sum = image_att + state_att
        image_weight = image_att / att_sum
        state_weight = state_att / att_sum

        # Apply the attention weights
        attended_image_output = image_output * image_weight  # Shape: [batch_size, 256]
        attended_state_output = state_output * state_weight  # Shape: [batch_size, 64]

        # Concatenate attended outputs
        combined_features = th.cat([attended_image_output, attended_state_output], dim=1)  # Shape: [batch_size, 320]

        # Final layer for combined features
        combined_features = self.fc_combined(combined_features)  # Shape: [batch_size, 320]

        return combined_features


class MyFeatureExtractorLidar(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # 输入给net_arch网络的特征维度=图像特征维度+自行车的状态向量维度
        super().__init__(observation_space, features_dim=128)

        self.lidar_model = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 23, 128),
            nn.ReLU(),
            nn.Linear(128, 96),
            nn.ReLU(),
        )

        # 定义状态特征提取器
        self.state_model = nn.Sequential(
            nn.Linear(5, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),  # 增加一层 Linear
            nn.LeakyReLU(0.2),
        )

        # Feature fusion with attention
        self.fusion_layer = AttentionFusion(lidar_dim=96, state_dim=32, output_dim=128)

    def forward(self, observations) -> th.Tensor:
        lidar_obs = observations["lidar"]
        lidar_obs = lidar_obs.clone().detach().float()
        lidar_obs = lidar_obs.unsqueeze(1)
        lidar_feat = self.lidar_model(lidar_obs)  # 通过图像特征提取器处理图像特征

        obs = observations["obs"]
        state_feat = self.state_model(obs)

        combined_features = self.fusion_layer(lidar_feat, state_feat)
        return combined_features


class AttentionFusion(nn.Module):
    def __init__(self, lidar_dim, state_dim, output_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(lidar_dim + state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.fc = nn.Linear(lidar_dim + state_dim, output_dim)

    def forward(self, lidar_features, state_features):
        combined = th.cat([lidar_features, state_features], dim=1)
        attention_weights = self.attention(combined)
        fused_features = attention_weights * combined
        return self.fc(fused_features)