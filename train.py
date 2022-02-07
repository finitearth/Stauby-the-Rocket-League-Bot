import rlgym
import torch.cuda
from rlgym.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from obs_builder import CustomObsBuilder, PolarObsBuilder
from action_parser import CustomActionParser
from reward import Reward, NectoRewardFunction
from actor_critic import ActorCritic
from state_setter import StateSetter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

obs_builder = PolarObsBuilder() #CustomObsBuilder
action_parser = CustomActionParser()
obs_space = obs_builder.obs_dim
action_space = action_parser.action_dim
state_setter = StateSetter()

# HyperParams
reward = Reward()#NectoRewardFunction()
lr = 1e-4
log_std_init = 0.
policy_kwargs = {
    "arch": {
        "v":  [obs_space, 4, 4, 1],
        "pi": [obs_space, 4, 4,  action_space]
    },
    "log_std_init": log_std_init
}

env = rlgym.make(
    reward_fn=reward,
    game_speed=80,
    terminal_conditions=(NoTouchTimeoutCondition(500), GoalScoredCondition()),
    self_play=False,
    state_setter=state_setter,
    action_parser=action_parser,
    obs_builder=obs_builder,
    use_injector=False,
    force_paging=False
    )

model = PPO(
    ActorCritic,
    # n_steps=2048,
    env=env,
    policy_kwargs=policy_kwargs,
    learning_rate=lr,
    device=device,
    verbose=10
    )

callback = EvalCallback(env, best_model_save_path="models/v0_small", deterministic=False)
model.learn(total_timesteps=1e7, callback=callback)
model.save("models/v0_2_1")
