import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体 ('SimHei') 或其他支持中文的字体
rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

data = [27560.14453125, 70164.9296875, 101.0]

x_pos = range(len(data))  # 定义柱子的位置，这里使用数据的索引作为位置
x_labels = ['基准组', '实验组1', '实验组2']
plt.bar(x_pos, data)  # 绘制柱状图

plt.xticks(x_pos, x_labels)  # 设置x轴标签
# plt.title('Simple Bar Chart')
# plt.xlabel('Index')
plt.ylabel('绝对值积分')
plt.savefig('bar.svg', format='svg')
plt.show()
