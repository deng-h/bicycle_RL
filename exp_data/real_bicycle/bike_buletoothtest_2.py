import serial
import time
import threading
import re
import logging
import csv
from datetime import datetime

# 配置串口
ser = serial.Serial(
    port='COM5',  # 根据实际情况修改端口号
    baudrate=115200,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    bytesize=serial.EIGHTBITS,
    timeout=1
)

# 全局变量
buffer = ""
data_ready = threading.Event()
lock = threading.Lock()

# 初始化日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def calculate_check(pidReceive):
    int_part = int(pidReceive)
    frac_part = int(round((pidReceive - int_part) * 100))
    return abs(int_part + frac_part)


def send_command(pidReceive):
    check = calculate_check(pidReceive)
    command = f"#{pidReceive},{check}*"
    with lock:
        ser.reset_input_buffer()
        ser.write(command.encode())
    logging.info(f"发送命令: {command}")


def parse_data0(data):
    try:
        # 使用正则表达式匹配数据格式（支持科学计数法）
        match = re.search(
            r'^#([+-]?\d+),'  # 第一段：整数（允许正负）
            r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?),'  # 第二段：浮点数（支持科学计数法）
            r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?),'
            r'([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\*$',  # 第四段：浮点数，以*结尾
            data.strip()  # 清理首尾空白字符
        )
        # print(data)
        # print(match)
        # print('----------------')
        if match:
            # 类型转换验证
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


# 串口数据接收线程
def serial_receiver():
    global buffer
    while True:
        if ser.in_waiting > 0:
            with lock:
                # 读取原始字节并追加到缓冲区
                raw_data = ser.read(ser.in_waiting)
                # print(f"==={raw_data}")
                buffer += raw_data.decode(errors='ignore').replace('\n', '')  # 去除换行符
                # 限制缓冲区大小
                if len(buffer) > 100:
                    buffer = buffer[-100:]
                data_ready.set()

def main():
    global buffer
    pid_data = []
    if ser.is_open:
        logging.info("串口已打开")
    else:
        logging.error("串口未打开")
        return

    # 启动串口接收线程
    receiver_thread = threading.Thread(target=serial_receiver, daemon=True)
    receiver_thread.start()

    try:
        while True:
            data_ready.wait()
            frames = []
            with lock:
                # 正则匹配所有完整帧
                # logging.info(f"buffer: {buffer}")
                frames = re.findall(r'#.*?\*', buffer)
                buffer = re.sub(r'#.*?\*', '', buffer)  # 移除已处理数据

            for frame in frames:
                # logging.info(f"接收帧: {frame}")
                encoder_value, roll_value, gyro_value, pid = parse_data0(frame)
                if encoder_value is not None and roll_value is not None and gyro_value is not None and pid is not None:
                    send_command(-125.0)
                    data = [encoder_value, roll_value, gyro_value, pid]
                    print(data)
                    pid_data.append(data)

            data_ready.clear()
            # time.sleep(0.01)  # 防止CPU占用过高
    except KeyboardInterrupt:
        logging.info("程序终止")
    finally:
        ser.close()
        now = datetime.now()
        formatted_time = now.strftime("%m%d_%H%M")
        logging.info("串口已关闭")
        with open(f"pid_data_{formatted_time}.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['encoder_value', 'roll_value', 'gyro_value', 'pid'])  # 写入标题行
            for value in pid_data:
                writer.writerow(value)  # 每个值写成一行


if __name__ == "__main__":
    main()
