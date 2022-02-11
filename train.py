import torch
from rlgym.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
from rlgym_tools.sb3_utils import SB3MultipleInstanceEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecMonitor
import wandb

from obs_builder import CustomObsBuilder, PolarObsBuilder
from action_parser import CustomActionParser
from reward import Reward, NectoRewardFunction
from actor_critic import ActorCritic, get_model
from state_setter import StateSetter
from match_creator import MatchCreator


if __name__ == "__main__":
    debug = 1
    if not debug: wandb.init(project="stauby", save_code=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parallel_matches = 1
    obs_builder = CustomObsBuilder()  #PolarObsBuilder()
    action_parser = CustomActionParser()
    obs_space = obs_builder.obs_dim
    action_space = action_parser.action_dim
    match_creator = MatchCreator(
        state_setter=StateSetter(),
        reward=Reward(),#NectoRewardFunction(),
        obs_builder=obs_builder,
        action_parser=action_parser,
        terminal_conds=(NoTouchTimeoutCondition(250), GoalScoredCondition()),
        game_speed=100
    )
    wandb.config = {
        "lr": 1e-3,
        "ex_arch": [obs_space, 32, 32, 8],
        "v_arch":  [8, 1],
        "pi_arch": [8, action_space],
        "activation": "tanh",
        "log_std_init": 0.,
        "optim": "SGD"
    }

    # HyperParams
    env = SB3MultipleInstanceEnv(match_creator.get_match, parallel_matches, force_paging=False, wait_time=30)
    env.reward_range = (-float("inf"), float("inf"))
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    env = VecMonitor(env)
    model = get_model(env, use_wandb=not debug)

    callback = EvalCallback(env, best_model_save_path="models/v0_small", deterministic=False, verbose=not debug)
    model.learn(total_timesteps=1e7, callback=callback)
    model.save("models/v0_2_1")
