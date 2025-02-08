import pandas as pd
import os

# 设置存放 CSV 文件的目录
input_folder = "D:\\data\\1-L\9-bicycle\\bicycle-rl\exp_data\平衡实验数据处理\\3"  # 替换为你的 CSV 文件所在目录
output_file = "D:\data\\1-L\9-bicycle\\bicycle-rl\exp_data\平衡实验数据处理\\3\\total.csv"  # 最终汇总的 CSV 文件名

# 创建一个空的 DataFrame 来存储汇总结果
combined_df = pd.DataFrame()

# 遍历目录下的所有 CSV 文件
for file_name in os.listdir(input_folder):
    if file_name.endswith('.csv'):  # 确保只处理 CSV 文件
        file_path = os.path.join(input_folder, file_name)
        # 读取 CSV 文件并跳过第一行
        df = pd.read_csv(file_path, skiprows=1)
        # 将当前文件内容追加到汇总 DataFrame
        combined_df = pd.concat([combined_df, df], ignore_index=True)

# 将汇总后的 DataFrame 保存为一个新的 CSV 文件
combined_df.to_csv(output_file, index=False, encoding='utf-8-sig')

print(f"所有CSV文件已成功汇总到 {output_file}")
