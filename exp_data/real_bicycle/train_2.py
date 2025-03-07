import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import os
import time

# 1. 数据读取和转换
# -----------------------
# 读取CSV文件
file_path = 'pid_data_0305_1429_AAA.csv'
data = pd.read_csv(file_path)

# 重新组织数据：每三行合并成一行
rows = []
for i in range(0, len(data) - 2, 1):  # 每3行处理一次
    row = [
        data.iloc[i]['roll_value'], data.iloc[i]['gyro_value'], data.iloc[i]['pid'],
        data.iloc[i+1]['roll_value'], data.iloc[i+1]['gyro_value'], data.iloc[i+1]['pid'],
        data.iloc[i+2]['roll_value'], data.iloc[i+2]['gyro_value'],
        data.iloc[i+2]['pid']  # 第3行的pid作为输出
    ]
    rows.append(row)

# 转换为DataFrame
processed_data = pd.DataFrame(rows, columns=[
    'roll_value_1', 'gyro_value_1', 'pid_1',
    'roll_value_2', 'gyro_value_2', 'pid_2',
    'roll_value_3', 'gyro_value_3',
    'pid'
])

# 数据归一化（标准化处理）
X = processed_data.iloc[:, :-1].values  # 输入（前6列）
y = processed_data.iloc[:, -1].values   # 输出（最后1列）

# 2. 数据归一化
# -----------------------
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

y_mean = y.mean()
y_std = y.std()
y = (y - y_mean) / y_std

# 3. 划分训练集和测试集
# -----------------------
train_size = int(0.99 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:] # 修正了原始代码中的笔误，y_test 正确赋值

# 4. 转换为PyTorch张量和创建DataLoader
# -----------------------
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 64
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 5. MLP模型定义 (已调整输入层维度为 6)
# -----------------------
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(8, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.model(x)

model = MLP()
model.to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 6. 训练循环
# -----------------------
num_epochs = 1000 # 训练轮数，您可以根据需要调整
for epoch in range(num_epochs):
    model.train()
    loss = 0.0
    for batch_idx, (data, targets) in enumerate(train_loader):
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    if epoch % 50 == 0:
        # 保存模型
        torch.save(model.state_dict(), f'model2_{epoch+1}.pth')

# 测试模型
model.eval()
total_test_loss = 0 # 用于累加所有批次的 loss
with torch.no_grad():
    for batch_idx, (data, targets) in enumerate(test_loader): # 遍历 test_loader 获取批次数据
        data, targets = data.to(device), targets.to(device)  # 将数据和目标移动到设
        predictions = model(data)
        test_loss = criterion(predictions, targets)
        total_test_loss += test_loss.item() # 累加每个批次的 loss
    average_test_loss = total_test_loss / len(test_loader) # 计算平均 loss
    print(f'Test Loss: {average_test_loss:.4f}') # 输出平均 loss

torch.save(model.state_dict(), 'model_AAA.pth')
#测试保存的模型
model = MLP()
model.load_state_dict(torch.load('model_AAA.pth'))
model.eval()
total_test_loss_loaded_model = 0
with torch.no_grad():
    for batch_idx, (data, targets) in enumerate(test_loader):
        predictions = model(data)
        test_loss = criterion(predictions, targets)
        total_test_loss_loaded_model += test_loss.item()
    average_test_loss_loaded_model = total_test_loss_loaded_model / len(test_loader)
    print(f'Test Loss (Loaded Model): {average_test_loss_loaded_model:.4f}')