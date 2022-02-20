import gym
from rlgym.utils.action_parsers import ContinuousAction
import numpy as np


class CustomActionParser(ContinuousAction):
    def __init__(self):
        super().__init__()
        self.action_dim = 2

    def get_action_space(self):
        return gym.spaces.Box(-float("inf"), float("inf"), shape=(self.action_dim,))

    def parse_actions(self, action_logits, state):
        action_logits[..., 1] *= 0.5
        actions = np.tanh(action_logits).reshape(-1, 2)
        filled_action = np.zeros((actions.shape[0], 8))
        filled_action[..., [0, 1]] = actions

        return filled_action


class DiscreteActionParser(CustomActionParser):
    def __init__(self):
        super().__init__()
        self.action_dim = 5

    def get_action_space(self):
        return gym.spaces.MultiDiscrete([2, 3])

    def parse_actions(self, action_logits, state):
        filled_action = np.zeros((action_logits.shape[0], 8))
        filled_action[..., 0] = action_logits[..., 0]
        filled_action[..., 1] = action_logits[..., 1] - 1

        return filled_action


