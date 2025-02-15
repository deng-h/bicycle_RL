import heapq


def heuristic(a, b):
    """
    曼哈顿距离启发式函数。
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def a_star_pathfinding(grid_map, start, goal):
    """
    A* 路径搜索算法。

    参数:
        grid_map: 二维网格地图 (NumPy 数组)
        start: 起始网格坐标 (元组或列表，例如 (0, 0))
        goal: 目标网格坐标 (元组或列表，例如 (10, 10))

    返回:
        如果找到路径，返回路径节点坐标列表 (从起点到终点)；否则返回 None。
    """
    rows, cols = grid_map.shape
    if not (0 <= start[0] < rows and 0 <= start[1] < cols and
            0 <= goal[0] < rows and 0 <= goal[1] < cols):
        raise ValueError("起始点或目标点超出网格地图范围")

    if grid_map[start[0], start[1]] == 1 or grid_map[goal[0], goal[1]] == 1:
        return None  # 起始点或目标点是障碍物，无法生成路径

    open_set = []  # 开放集合 (优先队列)
    heapq.heappush(open_set, (0, start)) # 将起点加入开放集合，优先级为 0

    came_from = {}  # 记录每个节点的前一个节点，用于路径回溯
    g_score = {start: 0}  # 从起点到每个节点的实际代价
    f_score = {start: heuristic(start, goal)} # 从起点到每个节点的估计代价 (f = g + h)

    while open_set:
        _, current = heapq.heappop(open_set) # 从开放集合中取出 f_score 最小的节点

        if current == goal:
            # 找到目标节点，回溯路径
            path = []
            node = goal
            while node in came_from:
                path.append(node)
                node = came_from[node]
            path.append(start)
            return path[::-1]  # 反转路径，使其从起点到终点

        # 遍历邻居节点 (上、下、左、右)
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            neighbor = (current[0] + dr, current[1] + dc)

            if not (0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols):
                continue  # 邻居节点超出网格边界
            if grid_map[neighbor[0], neighbor[1]] == 1:
                continue  # 邻居节点是障碍物

            temp_g_score = g_score[current] + 1  # 假设移动到邻居节点的代价为 1

            if neighbor not in g_score or temp_g_score < g_score[neighbor]:
                # 发现更优的路径或者第一次发现该邻居节点
                came_from[neighbor] = current
                g_score[neighbor] = temp_g_score
                f_score[neighbor] = temp_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor)) # 将邻居节点加入开放集合

    return None  # 未找到路径