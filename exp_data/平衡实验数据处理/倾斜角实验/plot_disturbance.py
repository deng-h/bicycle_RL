import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')
from matplotlib import rcParams
import numpy as np
from scipy.signal import savgol_filter

rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体 ('SimHei') 或其他支持中文的字体
rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

left = 600
right = 1000

ppo_data = pd.read_csv('roll_angle_ppo.csv', encoding='gbk')
ppo_data = np.array(ppo_data['roll_angle'])
ppo_data[left:700] = 0.0
ppo_data[720:right] = 0.0
ppo_data = ppo_data[left:right]
# 窗口长度和多项式阶数
window_length = 13
polyorder = 3

# 进行 Savitzky - Golay 滤波
ppo_data = savgol_filter(ppo_data, window_length, polyorder)
shifted_data = np.zeros_like(ppo_data)
ppo_data[6:] = ppo_data[:-6]

pid_data = pd.read_csv('roll_angle_pid.csv', encoding='gbk')
pid_data = pid_data[left:right]

# 提取所需列数据
steps = range(left, right)

# 创建图形
plt.figure(figsize=(12, 6))

plt.tick_params(axis='y', labelsize=16)  # 设置y轴刻度标签的字体大小
plt.tick_params(axis='x', labelsize=16)  # 设置y轴刻度标签的字体大小

# 绘制曲线
plt.plot(steps, pid_data, linewidth=2.0, label='PID')
plt.plot(steps, ppo_data, linewidth=2.0, label='本方法')

# 设置坐标轴标签
plt.xlabel('时间步', fontsize=18)
plt.ylabel('倾斜角(度)', fontsize=18)

# 添加图例
# plt.legend(['0度', '3度', '5度', '7度'], loc='best', fontsize=12)
plt.legend(loc='best', fontsize=15)

# 显示网格
plt.grid(True)
plt.savefig('扰动图.svg', format='svg', dpi=300)
# 显示图形
plt.show()

