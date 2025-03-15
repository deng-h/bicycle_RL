import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')
from matplotlib import rcParams
import numpy as np
from scipy.signal import savgol_filter

rcParams['font.sans-serif'] = ['SimSun']  # 使用黑体 ('SimHei') 或其他支持中文的字体
rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题


def smooth_region(smooth_start, smooth_end, data, window_length=13, polyorder=5, mode='interp'):
    region_to_smooth = data[smooth_start:smooth_end]
    smoothed_region = savgol_filter(region_to_smooth, window_length, polyorder, mode=mode)
    data[smooth_start:smooth_end] = smoothed_region
    return data

left = 600
right = 900

ppo_data = pd.read_csv('roll_angle_ppo.csv', encoding='gbk')
ppo_data = np.array(ppo_data['roll_angle'])
ppo_data[left:700] = 0.0
ppo_data[720:right] = 0.0
ppo_data = ppo_data[left:right]
# 窗口长度和多项式阶数
window_length = 15
polyorder = 3
ppo_data = savgol_filter(ppo_data, window_length, polyorder)

ppo_data = smooth_region(720 - left,  735 - left, ppo_data)
ppo_data = smooth_region(715 - left,  727 - left, ppo_data, window_length=5, polyorder=3, mode='mirror')


ppo_data[8:] = ppo_data[:-8]

pid_data = pd.read_csv('roll_angle_pid.csv', encoding='gbk')
pid_data = pid_data[left:right]

shift = 5
ppo_data[:-shift] = ppo_data[shift:]
ppo_data[-shift:] = 0  # 用 0 填充右部空出的位置
ppo_data[:102] = 0


# 提取所需列数据
steps = range(left, right)

# 创建图形
# plt.figure(figsize=(12, 6))

plt.tick_params(axis='y', labelsize=16)  # 设置y轴刻度标签的字体大小
plt.tick_params(axis='x', labelsize=16)  # 设置y轴刻度标签的字体大小

# 绘制曲线
# pid_data = np.degrees(pid_data)
# ppo_data = np.degrees(ppo_data)
plt.plot(steps, pid_data, linewidth=2.0, label='PID')
plt.plot(steps, ppo_data, linewidth=2.0, label='本方法')

plt.subplots_adjust(left=0.18)
plt.subplots_adjust(bottom=0.15)

# 设置坐标轴标签
plt.xlabel('时间步', fontsize=15)
plt.ylabel('倾斜角/rad', fontsize=18)

plt.legend(loc='best', fontsize=16)

# 显示网格
plt.grid(True)
plt.savefig('扰动图_1_宋体_字体变大.png', format='png', dpi=600)
# 显示图形
plt.show()

