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
    # logging.info(f"发送命令: {command}")


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

    # 启动串口接收线程
    receiver_thread = threading.Thread(target=serial_receiver_2, daemon=True)
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
                index, roll_value, gyro_value, pid = parse_data0(frame)
                if index is not None and roll_value is not None and gyro_value is not None and pid is not None:
                    send_command(pid)
                    data = [index, roll_value, gyro_value, pid]
                    pid_data.append(data)
                    if index % 1000 == 0:
                        print(data)

            data_ready.clear()
            time.sleep(0.01)  # 防止CPU占用过高
    except KeyboardInterrupt:
        logging.info("程序终止")
    finally:
        ser.close()
        now = datetime.now()
        formatted_time = now.strftime("%m%d_%H%M")
        logging.info("串口已关闭")
        with open(f"pid_data_{formatted_time}.csv", mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['index', 'roll_value', 'gyro_value', 'pid'])  # 写入标题行
            for value in pid_data:
                writer.writerow(value)  # 每个值写成一行


if __name__ == "__main__":
    main()
