import pybullet as p
import os


class Wall:
    def __init__(self, client, base):
        f_name = os.path.join(os.path.dirname(__file__), 'wall.xml')
        self.Id = p.loadURDF(fileName=f_name,
                   basePosition=[base[0], base[1], 0],
                   physicsClientId=client)


