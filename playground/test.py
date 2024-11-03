import gymnasium
import numpy as np
action_space = gymnasium.spaces.box.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
action_space2 = gymnasium.spaces.box.Box(
            low=np.array([-1.0, -2.0, -1.0]),
            high=np.array([1.0, 2.0, 1.0]),
            shape=(3,),
            dtype=np.float32)

print(action_space)
print(action_space2)