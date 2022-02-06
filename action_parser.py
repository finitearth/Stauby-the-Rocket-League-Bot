import gym
from rlgym.utils.action_parsers import ContinuousAction
import numpy as np


class CustomActionParser(ContinuousAction):
    def __init__(self):
        super().__init__()

    def get_action_space(self):
        return gym.spaces.Box(-1, 1, shape=(2,))

    def parse_actions(self, actions, state):
        actions = actions.reshape(-1, 2)
        filled_action = np.zeros((actions.shape[0], 8))
        filled_action[..., [0, 1]] = actions#[..., 0], actions[..., 1]


        return filled_action
