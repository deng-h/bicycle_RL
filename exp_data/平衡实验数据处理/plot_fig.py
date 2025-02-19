import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体 ('SimHei') 或其他支持中文的字体
rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

filename = "./ZBicycleBalanceEnv-v0_4/total.csv"
data = pd.read_csv(filename)

data = data.sort_values(by='t')  # 根据 't' 列升序排序
r_values = data['r']  # 提取 'r' 值
n = len(r_values)  # 获取数据长度
interval = 14  # 设置采样间隔

# 采样数据
sampled_indices = range(0, n, interval)
sampled_r = r_values.iloc[sampled_indices]
x_values = list(sampled_indices)  # 生成横坐标

# 绘制折线图
# plt.figure(figsize=(13, 6))
plt.plot(x_values, sampled_r, '-', linewidth=1.1)
plt.xlabel('回合数/轮', fontsize=15)
plt.ylabel('奖励值', fontsize=15)
# plt.title('Line Plot of r Values with Sampling')
# plt.grid()

# plt.savefig('line_plot_r_values.bmp')
plt.savefig('line.svg', format='svg')

# 显示图像
plt.show()
