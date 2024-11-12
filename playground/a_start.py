import json
import numpy as np
from heapq import heappop, heappush


# 读取地图文件并生成地图布局
def load_maze(maze_config):
    with open(maze_config, 'r') as file:
        data = json.load(file)
    maze_layout = np.array(data['maze'])
    return maze_layout


# 查找起点和终点位置
def find_start_goal(maze_layout):
    start, goal = None, None
    for y, row in enumerate(maze_layout):
        for x, cell in enumerate(row):
            if cell == 'S':
                start = (y, x)
            elif cell == 'G':
                goal = (y, x)
    return start, goal


# A*算法实现
def a_star(maze_layout, start, goal):
    def heuristic(a, b):
        # 使用曼哈顿距离作为启发式估计
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        _, current = heappop(open_set)

        if current == goal:
            # 生成路径
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        y, x = current
        # 遍历邻居节点（上下左右四个方向）
        for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            neighbor = (y + dy, x + dx)

            # 检查邻居是否在地图范围内，并且不是障碍物
            if (0 <= neighbor[0] < maze_layout.shape[0] and
                    0 <= neighbor[1] < maze_layout.shape[1] and
                    maze_layout[neighbor[0], neighbor[1]] != 'X'):

                tentative_g_score = g_score[current] + 1

                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))

    return None  # 若无路径可达终点则返回None


# 主函数：加载地图、执行A*算法并输出路径
def main(maze_config):
    maze_layout = load_maze(maze_config)
    print(maze_layout)
    start, goal = find_start_goal(maze_layout)

    if start is None or goal is None:
        print("无法找到起点或终点！")
        return

    path = a_star(maze_layout, start, goal)
    if path:
        print("找到路径：", path)
    else:
        print("没有可行路径！")


# 调用主函数，传入地图文件路径
main('D:\\data\\1-L\\9-bicycle\\bicycle-rl\\bicycle_dengh\\resources\\maze\\maze_config.json')
