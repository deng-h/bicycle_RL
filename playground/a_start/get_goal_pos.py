import random

def get_goal_pos():
    # (11, 20)
    # (20, 22)
    # (14, 26)
    # (12, 10)
    # (26, 21)
    goal = [(11, 20), (20, 22), (14, 26), (12, 10), (26, 21)]

    return random.choice(goal)
