import torch
from stable_baselines3.common.distributions import DiagGaussianDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn


class ActorCritic(ActorCriticPolicy):
    def __init__(self, obs_space, action_space, lr=10**-5, log_std_init=1, arch=None, use_sde=False, optim="Adam", activation=torch.nn, **kwargs):
        super(ActorCriticPolicy, self).__init__(obs_space, action_space, lr)
        self.clip_range = lambda x: .1
        self.clip_range_vf = lambda x: .1
        self.features_extractor = Net(arch=arch["ex"], activation=activation)
        self.action_net = Net(arch=arch["pi"], activation=activation)
        self.value_net = Net(arch=arch["v"], activation=activation)
        self.use_sde = use_sde
        self.action_dist = DiagGaussianDistribution(action_space.shape[0])
        self.log_std = nn.Parameter(torch.ones(action_space.shape[0]) * log_std_init, requires_grad=False)
        if optim == "Adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr(0), **self.optimizer_kwargs)
        elif optim == "SGD":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr(0), **self.optimizer_kwargs)
        print(f"Network set up:\n"
              f"extractor: {self.features_extractor}\n"   
              f"value net: {self.value_net}\n"
              f"policy net: {self.action_net}\n"
              f"GL HF! :)")

    def forward(self, obs, deterministic=False, use_sde=False):
        deterministic = True
        x = self.features_extractor(obs)
        values = self.value_net(x)
        mean_action = self.action_net(x)
        distribution = self.action_dist.proba_distribution(mean_action, self.log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        distribution = self.action_dist.proba_distribution(actions, self.log_std)
        log_prob = distribution.log_prob(actions)
        x = self.features_extractor(obs)
        values = self.value_net(x)

        return values, log_prob, distribution.entropy()

    def predict(self, obs, state=None, mask=None, deterministic=False):
        with torch.no_grad():
            obs = torch.Tensor(obs)
            x = self.features_extractor(obs)
            actions = self.action_net(x)

        return actions, state


class Net(nn.Module):
    def __init__(self, arch, activation):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(arch[i-1], arch[i])for i in range(len(arch))[1:]
            ]
        )
        self.activation = activation

    def forward(self, x):
        x = x.float()
        for layer in self.layers[:-1]:
            x = layer(x)
            x = self.activation(x)
        x = self.layers[-1](x)
        return x

