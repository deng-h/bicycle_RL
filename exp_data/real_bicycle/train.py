import pandas as pd
import numpy as np
import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader  # 导入TensorDataset和DataLoader

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
# X = data[['encoder_value', 'roll_value', 'gyro_value']].values # 您之前的代码包含 encoder_value，但后面注释掉了，所以这里沿用您注释掉的版本
X = data[['roll_value', 'gyro_value']].values
y = data['pid'].values

# 数据归一化
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

y_mean = y.mean()
y_std = y.std()
y = (y - y_mean) / y_std

# 划分训练集和测试集
train_size = int(0.99 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], X[train_size:] # 这里原始代码y_test 赋值成了X_test， 应该是笔误，已更正为 y_test = y[train_size:]
y_test = y[train_size:]

# 转换为PyTorch张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# ***  创建TensorDataset和DataLoader  ***
# 1. 创建训练集和测试集的TensorDataset
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# 检查 GPU 可用性并指定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 5120

# 3. 创建训练集和测试集的DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True) # 训练集通常需要打乱顺序
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False) # 测试集不需要打乱，顺序不影响测试结果

# 定义MLP回归模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(2, 128), # 增加第一层神经元
            nn.ReLU(),
            nn.Linear(128, 256), # 增加第二层神经元
            nn.ReLU(),
            nn.Linear(256, 512), # 增加第三层神经元
            nn.ReLU(),
            nn.Linear(512, 256), # 新增层
            nn.ReLU(),
            nn.Linear(256, 128), # 新增层
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        return self.model(x)

# 初始化模型、损失函数和优化器
model = MLP()
model.to(device) # 将模型移动到设备
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    loss = 0.0
    # ***  修改训练循环以使用DataLoader  ***
    for batch_idx, (data, targets) in enumerate(train_loader): # 遍历 train_loader 获取批次数据
        data, targets = data.to(device), targets.to(device) # 将数据和目标移动到设
        optimizer.zero_grad()
        outputs = model(data) # data 就是一个批次的输入 X， targets 是对应的批次标签 y
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}') # 注意这里的 Loss 是 **最后一个批次的 Loss**,  如果要看 **平均 Loss**， 需要在每个 epoch 累加所有批次的 loss 并求平均。

    if (epoch+1) % 50 == 0:
        # 保存模型
        torch.save(model.state_dict(), f'model_{epoch+1}.pth')

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


#保存模型
torch.save(model.state_dict(), 'model_yjk.pth')
#测试保存的模型
model = MLP()
model.load_state_dict(torch.load('model_yjk.pth'))
model.eval()
total_test_loss_loaded_model = 0
with torch.no_grad():
    for batch_idx, (data, targets) in enumerate(test_loader):
        predictions = model(data)
        test_loss = criterion(predictions, targets)
        total_test_loss_loaded_model += test_loss.item()
    average_test_loss_loaded_model = total_test_loss_loaded_model / len(test_loader)
    print(f'Test Loss (Loaded Model): {average_test_loss_loaded_model:.4f}')