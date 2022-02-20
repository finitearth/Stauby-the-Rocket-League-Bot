import torch
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import DiagGaussianDistribution, CategoricalDistribution, \
    MultiCategoricalDistribution
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn
from torch.distributions import Normal

from callbacks import CustomLogger


def get_model(env):
    model = PPO(
        DiscreteActorCritic,
        n_steps=4096,
        clip_range=wandb.config["clip_range"],
        batch_size=wandb.config["batch_size"],
        env=env,
        learning_rate=wandb.config["lr"],
        ent_coef=wandb.config["ent_coef"],
        policy_kwargs={
            "arch": {
                "v": wandb.config["v_arch"],
                "pi": wandb.config["pi_arch"],
                "ex": wandb.config["ex_arch"]
            },
            "log_std_init": wandb.config["log_std_init"],
            "optim": wandb.config["optim"],
            "activation": wandb.config["activation"]
        }
    )
    return model


class ActorCritic(ActorCriticPolicy):
    def __init__(self, obs_space, action_space, lr=10**-5, log_std_init=1, arch=None, use_sde=False, optim="Adam", activation=None, **kwargs):
        super(ActorCriticPolicy, self).__init__(obs_space, action_space, lr)
        if activation == "tanh":
            activation = torch.tanh
        elif activation == "lrelu":
            activation = torch.nn.LeakyReLU()
        else:
            raise ValueError("_.-")
        self.features_extractor = Net(arch=arch["ex"], activation=activation)
        self.action_net = Net(arch=arch["pi"], activation=activation)
        self.value_net = Net(arch=arch["v"], activation=activation)
        self.use_sde = use_sde
        self.action_dist = DiagGaussianDistribution(action_space.shape[0])
        self.log_std = nn.Parameter(torch.ones(action_space.shape[0]) * log_std_init, requires_grad=True)#False
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
        x = self.features_extractor(obs)
        values = self.value_net(x)
        mean_action = self.action_net(x)
        distribution = self.action_dist.proba_distribution(mean_action, self.log_std)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        x = self.features_extractor(obs)
        values = self.value_net(x)
        new_mean_action = self.action_net(x)
        new_distribution = self.action_dist.proba_distribution(new_mean_action, self.log_std)
        new_log_prob = new_distribution.log_prob(actions)

        return values, new_log_prob, new_distribution.entropy()

    def predict(self, obs, state=None, mask=None, deterministic=True):
        with torch.no_grad():
            obs = torch.Tensor(obs)
            x = self.features_extractor(obs)
            actions = self.action_net(x)

        return actions, state


class DiscreteActorCritic(ActorCritic):
    def __init__(self, obs_space, action_space, lr=10**-5, log_std_init=1, arch=None, use_sde=False, optim="Adam", activation=None, **kwargs):
        super().__init__(obs_space, action_space, lr=lr, log_std_init=log_std_init, arch=arch, use_sde=use_sde, optim=optim, activation=activation, **kwargs)
        self.action_dist = MultiCategoricalDistribution([2, 3])

    def forward(self, obs, deterministic=False, use_sde=False):
        x = self.features_extractor(obs)
        values = self.value_net(x)
        mean_action = self.action_net(x)
        distribution = self.action_dist.proba_distribution(action_logits=mean_action)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        return actions, values, log_prob

    def evaluate_actions(self, obs, actions):
        x = self.features_extractor(obs)
        values = self.value_net(x)
        action_logits = self.action_net(x)
        distribution = self.action_dist.proba_distribution(action_logits=action_logits)
        log_prob = distribution.log_prob(actions)

        return values, log_prob, distribution.entropy()

    def predict(self, obs, state=None, mask=None, deterministic=True):
        with torch.no_grad():
            obs = torch.Tensor(obs)
            x = self.features_extractor(obs)
            actions = self.action_net(x)
            actions = actions.split([2, 3])
            actions = [a.argmax(dim=1) for a in actions]

        return actions, state


class Net(nn.Module):
    def __init__(self, arch, activation):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(arch[i-1], arch[i]) for i in range(len(arch))[1:]
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


