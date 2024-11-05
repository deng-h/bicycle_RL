import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import platform


def plot_csv_column(csv_path, column_name):
    df = pd.read_csv(csv_path, skiprows=1)

    # 获取指定列的数据
    column_data = df[column_name]

    # 计算数据的最大值和最小值，并设置上下余量的比例
    margin_percentage = 0.1  # 这里设置余量比例为10%，可根据需求调整
    data_max = column_data.max()
    data_min = column_data.min()
    upper_margin = data_max * margin_percentage
    lower_margin = data_min * margin_percentage

    # 设置坐标轴范围，上下留有余量
    plt.ylim(data_min - np.abs(lower_margin), data_max + np.abs(upper_margin))

    # 绘制指定列的数据图
    plt.plot(df[column_name])
    plt.xlabel('epochs')
    plt.ylabel(column_name)
    plt.title(f'{column_name} Data')
    system = platform.system()
    if system == "Windows":
        plt.show()
    elif system == "Linux":
        plt.savefig(f'{column_name}_plot.png')
        plt.close()


if __name__ == '__main__':
    plot_csv_column("./bicycle-rl/logs/ppo/BicycleMaze-v0_1/0.monitor.csv", "l")