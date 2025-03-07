import pandas as pd
import numpy as np
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim

# 获取所有CSV文件的路径
csv_files = glob.glob(os.path.join('./bicycle_pid', '*.csv'))

# 读取并合并所有CSV文件的数据
data_list = []
for file in csv_files:
    data = pd.read_csv(file)
    data_list.append(data)

# 合并所有数据
data = pd.concat(data_list, ignore_index=True)

# 提取输入和标签
X = data[['encoder_value', 'roll_value', 'gyro_value']].values
y = data['pid'].values

# 数据归一化
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

y_mean = y.mean()
y_std = y.std()
y = (y - y_mean) / y_std

# 划分训练集和测试集
train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# 定义MLP回归模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

criterion = nn.MSELoss()

#测试保存的模型
model = MLP()
model.load_state_dict(torch.load('model_yjk.pth'))
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    test_loss = criterion(predictions, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')


