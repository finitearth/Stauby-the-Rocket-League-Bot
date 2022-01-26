import torch
from stable_baselines3.ppo import PPO


class Agent:
    def __init__(self):
        self.model = PPO.load("src/v0_1")

    @torch.no_grad()
    def act(self, obs):
        action, _ = self.model.predict(obs)
        return action
