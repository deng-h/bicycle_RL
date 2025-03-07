import csv

import serial
import threading
import logging
import re
import glob
import os
import pandas as pd
import torch
import torch.nn as nn
from datetime import datetime
import time

# 串口配置 (从文件1)
ser = serial.Serial(
    port='COM5',
    baudrate=115200,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1
)
buffer = ""
data_ready = threading.Event()
lock = threading.Lock()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 数据预处理参数 (从文件2)
csv_files = glob.glob(os.path.join('./bicycle_pid', '*.csv'))
data_list = []
for file in csv_files:
    data = pd.read_csv(file)
    data_list.append(data)
data = pd.concat(data_list, ignore_index=True)
# X = data[['encoder_value', 'roll_value', 'gyro_value']].values
X = data[['roll_value', 'gyro_value']].values
y = data['pid'].values
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
y_mean = y.mean()
y_std = y.std()

# 模型定义 (从文件2)
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        # 对应模型model_1000_.pth
        # self.model = nn.Sequential(
        #     nn.Linear(2, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 128),
        #     nn.ReLU(),
        #     nn.Linear(128, 1)
        # )

        self.model = nn.Sequential(
            nn.Linear(2, 128), # 增加第一层神经元
            nn.ReLU(),
            nn.Linear(128, 256), # 增加第二层神经元
            nn.ReLU(),
            nn.Linear(256, 512), # 增加第三层神经元
            nn.ReLU(),
            nn.Linear(512, 256), # 新增层
            nn.ReLU(),
            nn.Linear(256, 128), # 新增层
            nn.ReLU(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.model(x)

model = MLP()
# model.load_state_dict(torch.load('model_1000_.pth')) # 加载模型
model.load_state_dict(torch.load('model_39500.pth')) # 加载模型
model.eval() # 设置为评估模式 (从文件2)

# 校验和计算函数 (从文件1)
def calculate_check(pidReceive):
    int_part = int(pidReceive)
    frac_part = int(round((pidReceive - int_part) * 100))
    return abs(int_part + frac_part)

# 发送命令函数 (从文件1)
def send_command(pidReceive):
    check = calculate_check(pidReceive)
    command = f"#{pidReceive},{check}*"
    with lock:
        ser.reset_input_buffer()
        ser.write(command.encode())
    # logging.info(f"发送命令: {command}")

# 数据解析函数 (从文件1)
def parse_data0(data):
    try:
        match = re.search(
            r'^#([+-]?\d+),'  # 第一段：整数（允许正负）
            r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?),'  # 第二段：浮点数（支持科学计数法）
            r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?),'
            r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\*$',  # 第四段：浮点数，以*结尾
            data.strip()  # 清理首尾空白字符
        )
        if match:
            encoder = int(match.group(1))
            rol = float(match.group(2))
            gyro = float(match.group(3))
            pid = float(match.group(4))
            return encoder, rol, gyro, pid
    except ValueError as e:
        logging.error(f"数值转换失败: {e} -> 原始数据: {data}")
    except Exception as e:
        logging.error(f"解析异常: {e} -> 原始数据: {data}")
    return None, None, None, None

# 串口数据接收线程 (从文件1)
def serial_receiver():
    global buffer
    while True:
        if ser.in_waiting > 0:
            with lock:
                # 读取原始字节并追加到缓冲区
                raw_data = ser.read(ser.in_waiting)
                print(f"==={raw_data}")
                buffer += raw_data.decode(errors='ignore').replace('\n', '')  # 去除换行符
                # 限制缓冲区大小
                if len(buffer) > 200: # 增加缓冲区大小，以容纳更多数据帧
                    buffer = buffer[-200:]
                data_ready.set()

def serial_receiver_2():
    global buffer
    temp_buffer = ""
    while True:
        if ser.in_waiting > 0:
            byte_data = ser.read(ser.in_waiting) #  一次性读取所有可用字节，提高效率
            try:
                byte_str = byte_data.decode('ascii', errors='ignore') # 使用 errors='ignore' 忽略无法解码的字节
                temp_buffer += byte_str
            except UnicodeDecodeError as e:
                logging.warning(f"UnicodeDecodeError: {e}, 忽略无法解码的字节")
                continue  # 忽略本次循环，继续接收

            frames = []
            start_index = 0
            while True:
                start_frame_index = temp_buffer.find('#', start_index) # 从上次查找位置开始找 '#'
                if start_frame_index == -1: # 找不到 '#'，说明没有新帧开始，退出内循环
                    break

                end_frame_index = temp_buffer.find('*', start_frame_index) # 从 '#' 位置开始找 '*'
                if end_frame_index == -1: # 找到 '#' 但找不到 '*'，说明帧不完整，跳出内循环，等待更多数据
                    break

                frame = temp_buffer[start_frame_index:end_frame_index+1] # 截取一个完整帧 (包含 # 和 *)
                # **更严格的帧内容校验**
                if is_valid_frame_content(frame): #  调用新的校验函数
                    frames.append(frame)
                    logging.debug(f"接收到有效帧: {frame}") # 使用 debug 级别日志
                # else:
                #     logging.warning(f"接收到无效帧 (内容校验失败): {frame}") # 记录无效帧

                start_index = end_frame_index + 1 # 更新下次查找的起始位置，跳过已处理的帧

            with lock:
                if frames: # 如果有有效帧
                    for frame in frames:
                        buffer += frame # 将有效帧添加到主缓冲区
                    data_ready.set() # 设置数据就绪标志
                temp_buffer = temp_buffer[start_index:] # 更新临时缓冲区，移除所有已处理的帧和无效帧片段

            # 限制临时缓冲区大小 (可选)
            if len(temp_buffer) > 400: #  可以适当增加临时缓冲区大小
                temp_buffer = temp_buffer[-400:]
            # time.sleep(0.001) # 可以考虑适当降低循环频率，减少 CPU 占用, 但不建议过度延迟，影响实时性

def is_valid_frame_content(frame):
    """
    校验帧内容是否符合预期格式
    """
    match = re.match(r'^#(-?\d+),(-?\d+(\.\d+)?),(-?\d+(\.\d+)?),(-?\d+(\.\d+)?)\*\s*$', frame) #  更精确的正则
    if not match:
        # logging.warning(f"帧格式正则匹配失败: {frame}")
        return False

    try:
        encoder = int(match.group(1))
        rol = float(match.group(2))
        gyro = float(match.group(4)) # 注意 group 索引，根据新的正则表达式调整
        pid = float(match.group(6))  # 注意 group 索引，根据新的正则表达式调整
        #  (可以添加更细致的数值范围校验，如果需要)
        return True # 格式和数值转换都成功，认为是有效帧

    except ValueError as e:
        # logging.error(f"帧数值转换失败: {e}, 帧内容: {frame}")
        return False
    except Exception as e:
        # logging.error(f"帧校验时发生未预料的异常: {e}, 帧内容: {frame}")
        return False

def main():
    global buffer
    pid_data = []
    if ser.is_open:
        logging.info("串口已打开")
    else:
        logging.error("串口未打开")
        return

    receiver_thread = threading.Thread(target=serial_receiver_2, daemon=True)
    receiver_thread.start()

    try:
        while True:
            data_ready.wait()
            frames = []
            with lock:
                frames = re.findall(r'#.*?\*', buffer)
                buffer = re.sub(r'#.*?\*', '', buffer)  # 移除已处理数据
            # print(frames)
            for frame in frames:
                encoder_value, roll_value, gyro_value, pid_ = parse_data0(frame)  # 原始数据pid改名为pid_下位机，避免混淆
                if encoder_value is not None and roll_value is not None and gyro_value is not None:
                    # print(f"encoder_value: {encoder_value}, roll_value: {roll_value}, gyro_value: {gyro_value}")
                    # 准备模型输入数据
                    # input_data = torch.tensor([[encoder_value, roll_value, gyro_value]], dtype=torch.float32)
                    input_data = torch.tensor([[roll_value, gyro_value]], dtype=torch.float32)

                    # 数据标准化，使用训练集上的均值和标准差
                    input_data_normalized = (input_data - torch.tensor(X_mean, dtype=torch.float32)) / torch.tensor(X_std, dtype=torch.float32)

                    # 模型预测
                    with torch.no_grad():
                        predicted_pid_normalized = model(input_data_normalized)

                    # 反标准化 (如果模型输出的是标准化后的PID，这里需要反标准化，但根据你的文件2, 模型输出训练目标是标准化后的pid，但是文件1 直接发送 -40.0 这样的原始pid值,  所以这里可能不需要反标准化，直接发送模型预测的标准化pid值对应的原始值即可。 假设下位机需要原始PID值，我们需要进行反标准化)
                    predicted_pid = predicted_pid_normalized * y_std + y_mean

                    #  这里假设下位机可以直接使用模型预测的标准化后的PID值, 如果下位机需要原始PID值，请取消注释反标准化的代码，并使用 predicted_pid
                    predicted_pid_to_send = predicted_pid.item()
                    predicted_pid_to_send = max(-800.0, min(800.0, predicted_pid_to_send))
                    predicted_pid_to_send = round(predicted_pid_to_send, 1)

                    pid_ = max(-800.0, min(800.0, pid_))
                    # 发送命令到下位机 (发送模型预测的PID值)
                    send_command(predicted_pid_to_send)

                    data = [encoder_value, roll_value, gyro_value, pid_, predicted_pid_to_send] # 记录原始pid 和 模型预测pid
                    pid_data.append(data)
                    time.sleep(0.001)
            data_ready.clear()
            # print(time.time())
            # print('----------------')

    except KeyboardInterrupt:
        logging.info("程序终止")
    finally:
        ser.close()
        now = datetime.now()
        formatted_time = now.strftime("%m%d_%H%M")
        logging.info("串口已关闭")
        with open(f"pid_data_{formatted_time}.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['encoder_value', 'roll_value', 'gyro_value', 'pid', 'predicted_pid_to_send'])  # 写入标题行
            for value in pid_data:
                writer.writerow(value)  # 每个值写成一行


if __name__ == "__main__":
    main()
