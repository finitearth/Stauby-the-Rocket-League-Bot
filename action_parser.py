import gym
from rlgym.utils.action_parsers import ContinuousAction
import numpy as np


class CustomActionParser(ContinuousAction):
    def __init__(self):
        super().__init__()
        self.action_dim = 2

    def get_action_space(self):
        return gym.spaces.Box(-1, 1, shape=(self.action_dim,))

    def parse_actions(self, action_logits, state):
        action_logits[..., 1] *= 0.1
        actions = np.tanh(action_logits).reshape(-1, 2)
        filled_action = np.zeros((actions.shape[0], 8))
        filled_action[..., [0, 1]] = actions

        return filled_action
