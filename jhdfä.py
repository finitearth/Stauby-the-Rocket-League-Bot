import numpy as np

actions = np.array([1, 2])


actions = actions.reshape(-1, 2)
filled_action = np.zeros((actions.shape[0], 8))
filled_action[:, [0, 1]] = actions[..., 0], actions[..., 1]
