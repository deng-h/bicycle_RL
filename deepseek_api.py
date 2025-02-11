import os
from openai import OpenAI

client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    api_key="sk-bb697fd598274bc781c01d20d33c6638", # 如何获取API Key：https://help.aliyun.com/zh/model-studio/developer-reference/get-api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
content = r'''
我的硕士论文主题是用强化学习的方法（PPO算法）控制无人自行车平衡（使用惯性轮控制平衡）、避障导航以及行驶到目标点。
我使用stable_baselines3强化学习框架做训练。
我的观测空间为observation_space，
observation_space = gymnasium.spaces.Dict({
	# lidar大小是360×1，为激光雷达扫描环境一周的数据
	"lidar": gymnasium.spaces.box.Box(low=0., high=150., shape=(360,), dtype=np.float32),
	# obs为翻滚角, 车把角度, 后轮速度, 车与目标点距离, 车与目标点角度
	"obs": gymnasium.spaces.box.Box(
		low=np.array([-math.pi, -1.57, -10., -100., -math.pi]),
		high=np.array([math.pi, 1.57, 10., 100., math.pi]),
		shape=(5,),
		dtype=np.float32
	),
})
我的动作空间是action_space，分别控制车把角度，前后轮速度, 飞轮速度
action_space = gymnasium.spaces.box.Box(
	low=np.array([-1.57, 0., -120.]),
	high=np.array([1.57, 5., 120.]),
	shape=(3,),
	dtype=np.float32)
MyFeatureExtractorLidar类是用于将观测空间的激光雷达数据和自行车数据做融合，即observation_space中的lidar和obs。
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
            nn.Linear(32, 32),  # 保持输出维度不变
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
我想问我的MyFeatureExtractorLidar设计的是否合理，请帮我优化MyFeatureExtractorLidar。
注意，不要拘泥于我给的MyFeatureExtractorLidar，你也可以定义自己的特征提取器，从而更好地完成给定任务。
注意，我的激光雷达数据和自行车状态数据在提取特征之前已经做过归一化了
'''

completion = client.chat.completions.create(
    model="deepseek-r1",  # 此处以 deepseek-r1 为例，可按需更换模型名称。
    messages=[
        {'role': 'user',
         'content': content}
        ]
)

# 通过reasoning_content字段打印思考过程
print("思考过程：")
print(completion.choices[0].message.reasoning_content)
# 通过content字段打印最终答案
print("最终答案：")
print(completion.choices[0].message.content)