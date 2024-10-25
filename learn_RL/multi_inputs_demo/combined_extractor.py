import gymnasium as gym
import torch as th
from torch import nn

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# 这个类的主要功能是接收多种类型的观测数据，针对每种类型设计不同的特征提取方式，并将提取后的特征拼接成一个统一的Tensor，供强化学习模型输入
# 这种设计适用于复杂的观测空间，例如同时包含图像和向量信息的环境
class CombinedExtractor(BaseFeaturesExtractor):
    # 这里传入的 observation_space 是一个 gym.spaces.Dict，代表了不同类型的观测空间。每个键（key）对应一种观测类型，如图像或向量
    def __init__(self, observation_space: gym.spaces.Dict):
        # 先初始化父类 BaseFeaturesExtractor，并且暂时设置 features_dim=1，因为我们还不知道最终的特征维度，需要通过遍历观测空间来确定
        super().__init__(observation_space, features_dim=1)
        extractors = {}
        total_concat_size = 0
        # 我们需要知道该提取器输出的大小，因此，遍历所有空间并计算输出特征大小并构建特征提取器
        for key, subspace in observation_space.spaces.items():
            if key == "image":
                # 这里假设图像是单通道（即灰度图像），首先用 MaxPool2d(4) 对图像进行下采样，减小图像的尺寸
                # 然后通过 Flatten() 将图像展平为一维向量，最后通过计算，更新拼接后的总特征维度 total_concat_size
                extractors[key] = nn.Sequential(nn.MaxPool2d(4), nn.Flatten())
                total_concat_size += subspace.shape[1] // 4 * subspace.shape[2] // 4
            elif key == "vector":
                # 对于向量类型的观测空间，使用一个全连接层 nn.Linear() 将输入向量映射到16维的特征空间，并更新 total_concat_size
                extractors[key] = nn.Linear(subspace.shape[0], 16)
                total_concat_size += 16

        # 特征提取器存储
        self.extractors = nn.ModuleDict(extractors)

        # 手动更新最终的特征维度，_features_dim是父类的成员变量
        self._features_dim = total_concat_size

    # 前向传播函数用于将输入的观测数据通过各个特征提取器，生成拼接后的特征向量，给后面的net_arch网络
    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            # 对于每个观测类型，调用对应的提取器对其进行处理，并将结果添加到 encoded_tensor_list 列表中
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        # 将所有提取到的特征张量在第1维（特征维度）进行拼接，形成一个统一的特征张量，供后续的强化学习算法使用
        return th.cat(encoded_tensor_list, dim=1)
