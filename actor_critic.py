import torch
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


class ActorCritic(ActorCriticPolicy):
    def __init__(self, obs_space, action_space, lr=10**-5, log_std_init=1, arch=None, use_sde=False, **kwargs):
        super(ActorCriticPolicy, self).__init__(obs_space, action_space, lr)
        self.action_net = Net(arch=arch["pi"])
        self.value_net = Net(arch=arch["v"])
        self.use_sde = use_sde
        self.action_dist = DiagGaussianDistribution(action_space.shape[0])
        self.log_std = nn.Parameter(torch.ones(action_space.shape[0]) * log_std_init, requires_grad=True)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr(0), **self.optimizer_kwargs)

    def forward(self, x, deterministic=False, use_sde=False):
        values = self.value_net(x)
        mean_action = self.action_net(x)
        distribution = self.action_dist.proba_distribution(mean_action, self.log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        # actions = torch.tanh(actions)

        return actions, values, log_prob

    def evaluate_actions(self, x, actions):
        distribution = self._get_action_dist_from_latent(x)
        log_prob = distribution.log_prob(actions)
        values = self.value_net(x)

        return values, log_prob, distribution.entropy()

    def predict(self, observation, state=None, mask=None, deterministic=False):
        with torch.no_grad():
            observation = torch.Tensor(observation)
            actions, _, _ = self.forward(observation, deterministic=deterministic)

        return actions, state


class Net(nn.Module):
    def __init__(self, arch):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(arch[i-1], arch[i])for i in range(len(arch))[1:]
            ]
        )

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = layer(x)
            x = torch.tanh(x)
        x = self.layers[-1](x)
        return x

