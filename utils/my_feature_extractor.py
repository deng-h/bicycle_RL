from torch import nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym
import torch as th


# 自定义的特征提取器
class MyFeatureExtractor1(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # 输入给net_arch网络的特征维度=图像特征维度+自行车的状态向量维度
        super().__init__(observation_space, features_dim=256+64+3)
        self.image_model = nn.Sequential(
            # height_in=128, width_in=128, CHW, 在线卷积池化公式计算器 http://www.sqflash.com/cal.html
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 16x64x64
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
        last_action = observations["last_action"]
        image_output = self.image_model(image_obs)  # 通过图像特征提取器处理图像特征
        state_output = self.state_model(obs)  # 通过状态特征提取器处理状态特征
        combined_features = th.cat([image_output, state_output, last_action], dim=1)  # 拼接图像特征和状态特征
        return combined_features


class MyFeatureExtractor2(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # 输入给net_arch网络的特征维度=图像特征维度+自行车的状态向量维度
        super().__init__(observation_space, features_dim=256+64)
        self.image_model = nn.Sequential(
            # height_in=128, width_in=128, CHW, 在线卷积池化公式计算器 http://www.sqflash.com/cal.html
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),  # 16x64x64
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
            nn.ReLU()
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
