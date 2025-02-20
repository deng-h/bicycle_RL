import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体 ('SimHei') 或其他支持中文的字体
rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块的问题

# 数据定义
labels = ['基准组', '实验组1', '实验组2']
balance_accuracy = [0.0025552710697, 0.0218383833056, 0.0171900208622]
balance_stability = [0.045130921603, 0.5865021715, 0.454128509651]
balance_energy_consumption = [32.756727404, 30.01755640, 58.83321837]

x = np.arange(len(labels))  # x轴位置
width = 0.6  # 增加柱子的宽度，使柱子更紧凑

# 创建画布和子图
fig, axes = plt.subplots(1, 3, figsize=(12, 5))  # 调整画布大小

# 平衡精度柱状图
axes[0].bar(x, balance_accuracy, width, color=['blue', 'orange', 'green'])
axes[0].set_title('平衡精度')
axes[0].set_xticks(x)
axes[0].set_xticklabels(labels)
# axes[0].set_ylabel('翻滚角平均绝对值')

# 平衡稳定性柱状图
axes[1].bar(x, balance_stability, width, color=['blue', 'orange', 'green'])
axes[1].set_title('平衡稳定性')
axes[1].set_xticks(x)
axes[1].set_xticklabels(labels)
# axes[1].set_ylabel('翻滚角角速度标准差')

# 平衡能耗柱状图
axes[2].bar(x, balance_energy_consumption, width, color=['blue', 'orange', 'green'])
axes[2].set_title('平衡能耗')
axes[2].set_xticks(x)
axes[2].set_xticklabels(labels)
# axes[2].set_ylabel('惯性轮角速度平均绝对值')

# 调整布局
# plt.tight_layout()

# 显示图像
plt.show()