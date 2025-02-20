import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体 ('SimHei') 或其他支持中文的字体
rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

x_labels = ['基准组', '实验组1', '实验组2']
balance_accuracy = [0.0025552710697, 0.0218383833056, 0.0171900208622]
balance_stability = [0.045130921603, 0.5865021715, 0.454128509651]
balance_energy_consumption = [32.756727404, 30.01755640, 58.83321837]

plt.figure(figsize=(4, 6))

x_pos = range(len(balance_energy_consumption))  # 定义柱子的位置，这里使用数据的索引作为位置
plt.bar(x_pos, balance_energy_consumption, width=0.6)  # 绘制柱状图

plt.xticks(x_pos, x_labels, fontsize=12)  # 设置x轴标签
# plt.xlabel('Index')
# plt.ylabel('平衡精度(翻滚角平均绝对值)', fontsize=16)
# plt.ylabel('平衡稳定性(翻滚角角速度标准差)', fontsize=16)
plt.ylabel('平衡能耗(惯性轮角速度平均绝对值)', fontsize=16)

plt.tight_layout()
plt.savefig('bar.svg', format='svg')
plt.show()
