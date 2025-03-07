import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('QtAgg')
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体 ('SimHei') 或其他支持中文的字体
rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 读取 CSV 文件
data = pd.read_csv('roll_angle_data - 副本.csv', encoding='gbk')

# 提取所需列数据
steps = data['Step']
# roll_angle_0 = data['Roll Angle 0 degree']
roll_angle_3 = data['Roll Angle 3 degree']
roll_angle_5 = data['Roll Angle 5 degree']
roll_angle_7 = data['Roll Angle 7 degree']

# 创建图形
plt.figure(figsize=(12, 6))

plt.tick_params(axis='y', labelsize=16)  # 设置y轴刻度标签的字体大小
plt.tick_params(axis='x', labelsize=16)  # 设置y轴刻度标签的字体大小

# 绘制曲线
# plt.plot(steps, roll_angle_0, 'k-', linewidth=1.5)
plt.plot(steps, roll_angle_3, 'r-', linewidth=1.5)
plt.plot(steps, roll_angle_5, 'g-', linewidth=1.5)
plt.plot(steps, roll_angle_7, 'b-', linewidth=1.5)

# 设置坐标轴标签
plt.xlabel('时间步', fontsize=18)
plt.ylabel('倾斜角(度)', fontsize=18)

# 添加图例
# plt.legend(['0度', '3度', '5度', '7度'], loc='best', fontsize=12)
plt.legend(['3度', '5度', '7度'], loc='best', fontsize=15)

# 显示网格
plt.grid(True)
plt.savefig('不同倾斜角对比图.svg', format='svg', dpi=300)
# 显示图形
plt.show()

