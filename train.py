import numpy as np
import rlgym
import torch
from rlgym.utils import ObsBuilder
from rlgym.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
import wandb

from callbacks import CustomEvalCallback, CustomLogger
from obs_builder import CustomObsBuilder, PolarObsBuilder
from action_parser import CustomActionParser, DiscreteActionParser
from reward import Reward, NectoRewardFunction
from actor_critic import ActorCritic, get_model
from state_setter import StateSetter
from match_creator import MatchCreator


if __name__ == "__main__":
    debug = 0
    if not debug: wandb.init(project="stauby", save_code=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parallel_matches = 1
    obs_builder = PolarObsBuilder()#CustomObsBuilder() #
    action_parser = DiscreteActionParser()#CustomActionParser()
    obs_space = obs_builder.obs_dim
    action_space = action_parser.action_dim
    match_creator = MatchCreator(
        state_setter=StateSetter(),
        reward=Reward(),#NectoRewardFunction(),
        obs_builder=obs_builder,
        action_parser=action_parser,
        terminal_conds=(NoTouchTimeoutCondition(750), GoalScoredCondition()),
        game_speed=100
    )
    wandb.config = {
        "lr": 5e-4,
        "ex_arch": [obs_space, 8, 8, 4],
        "v_arch":  [4, 1],
        "pi_arch": [4, action_space],
        "activation": "tanh",
        "log_std_init": 0.,
        "optim": "Adam",
        "batch_size": 32,
        "ent_coef": 0,#0.01,
        "clip_range": 0.2
    }

    env = SB3MultipleInstanceEnv(match_creator.get_match, parallel_matches, force_paging=False, wait_time=30)
    env.reward_range = (-float("inf"), float("inf"))
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=float("inf"), clip_reward=float("inf"))#100.0)
    env = VecMonitor(env)

    model = get_model(env)
    if not debug: model._logger = CustomLogger()

    callback = CustomEvalCallback(env=env, eval_env=env, best_model_save_path="models/v0_small")
    model.learn(total_timesteps=1e7, callback=callback)
    model.save("models/v0_2_1")
