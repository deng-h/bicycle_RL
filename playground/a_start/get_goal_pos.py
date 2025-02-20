import random

def get_goal_pos():
    # (11, 20)  左上的点
    # (20, 22)
    # (14, 26)
    # (12, 10)
    # (26, 21)
    # goal_1 = [(11, 20), (20, 22), (14, 26), (12, 10), (26, 21)]
    goal_2 = [(11, 23), (-11, 23), (-5, 23), (5, 23),
              (7, 16), (-7, 16), (0, 16),
              (3, 12), (-4, 12), (2, 10),
              (3, 8), (-5, 8),
              (-11, 9), (11, 8)]
    # 权重列表，与 goal_2 元素对应
    weights_goal_2 = [1, 1, 1, 1,
                      10, 10, 10,
                      10, 10, 10,
                      10, 10,
                      1, 1]

    # 使用 random.choices() 进行加权随机选择，k=1 表示抽取一个元素
    chosen_elements = random.choices(population=goal_2, weights=weights_goal_2, k=1)
    return chosen_elements[0]  # 返回抽取的元素，而不是包含元素的列表
