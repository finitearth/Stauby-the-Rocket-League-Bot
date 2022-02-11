from gym.spaces import Box
from stable_baselines3 import PPO
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import register_policy, ActorCriticPolicy, get_policy_from_name


def dingi(_):
    return 0
dings = get_policy_from_name(ActorCriticPolicy, "MlpPolicy")(Box(0, 1, shape=(2,)), Box(0, 1, shape=(2,)), dingi)
print(dings.features_extractor)
print(dings.policy_net)
print(dings.value_net)
print()
