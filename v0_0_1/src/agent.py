import torch
from stable_baselines3.ppo import PPO
import actor_critic


class CustomPPO(PPO):
    def _setup_model(self):
        super(PPO, self)._setup_model()


class Agent:
    def __init__(self):
        self.model = CustomPPO.load("src/best_model")

    @torch.no_grad()
    def act(self, obs):
        action, _ = self.model.predict(obs)
        return action
