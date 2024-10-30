import os
from datetime import datetime

i = []

while len(i) < 3:
    i.append(9)
    print(1)

# 获取当前脚本的绝对路径
current_file_path = os.path.abspath(__file__)
print(current_file_path)
# 获取当前脚本所在的目录
current_dir = os.path.dirname(current_file_path)
print(current_dir)
# 可以通过多次调用os.path.dirname来获取更高层的目录，假设项目根目录是当前脚本文件的上两级目录
root_dir = os.path.dirname(current_dir)
print(root_dir)
print(datetime.now().strftime("%m%d%H%M"))