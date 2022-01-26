import math

import gym
import numpy as np
import rlgym
import torch
from rlgym.utils import RewardFunction, ObsBuilder, common_values
from rlgym.utils.reward_functions.common_rewards import LiuDistanceBallToGoalReward, SaveBoostReward, EventReward, \
    VelocityPlayerToBallReward, FaceBallReward
from rlgym.utils.state_setters import RandomState
from rlgym.utils.terminal_conditions.common_conditions import NoTouchTimeoutCondition, GoalScoredCondition
from rlgym.utils.action_parsers.continuous_act import ContinuousAction
from stable_baselines3 import PPO


class Reward(RewardFunction):
    def __init__(self, own_goal=False):
        super().__init__()
        self.own_goal = own_goal
        self.liu = LiuDistanceBallToGoalReward()
        self.vel = VelocityPlayerToBallReward()
        self.boost = SaveBoostReward()
        self.goal = EventReward(team_goal=20., concede=0., touch=10., shot=2, save=0., demo=0.)
        self.driving_to_ball = FaceBallReward()

    def get_reward(self, player, state, previous_action):
        goal_reward = self.goal.get_reward(player, state, previous_action)
        liu_reward = self.liu.get_reward(player, state, previous_action)
        vel_reward = self.vel.get_reward(player, state, previous_action)
        boost_reward = self.boost.get_reward(player, state, previous_action)
        driving_reward = self.driving_to_ball.get_reward(player, state, previous_action)

        reward = goal_reward + .2 * liu_reward + .0 * boost_reward + .1 * driving_reward + .2 * vel_reward
        reward /= .5 * 250

        return reward

    def reset(self, initial_state):
        self.liu.reset(initial_state)
        self.vel.reset(initial_state)
        self.boost.reset(initial_state)
        self.goal.reset(initial_state)
        self.driving_to_ball.reset(initial_state)


class CustomActionParser(ContinuousAction):
    def __init__(self):
        super().__init__()

    def get_action_space(self):
        return gym.spaces.Box(-1, 1, shape=(3,))

    def parse_actions(self, actions, state):
        if len(actions.shape) == 1:
            filled_action = np.zeros(8)
            filled_action[[0, 1, 6]] = (actions[0] + 1) / 2, actions[1], actions[2] > 0

        else:
            filled_action = np.zeros((actions.shape[0], 8))
            # Throttle, Steer, Boost
            filled_action[..., [0, 1, 6]] = (actions[..., 0] + 1) / 2, actions[..., 1], actions[..., 2] > 0

        return filled_action


class CustomObsBuilder(ObsBuilder):
    def __init__(self, pos_coef=1 / 2300, ang_coef=1 / math.pi, lin_vel_coef=1 / 2300, ang_vel_coef=1 / math.pi):
        """
        :param pos_coef: Position normalization coefficient
        :param ang_coef: Rotation angle normalization coefficient
        :param lin_vel_coef: Linear velocity normalization coefficient
        :param ang_vel_coef: Angular velocity normalization coefficient
        """
        super().__init__()
        self.POS_COEF = pos_coef
        self.ANG_COEF = ang_coef
        self.LIN_VEL_COEF = lin_vel_coef
        self.ANG_VEL_COEF = ang_vel_coef

    def reset(self, initial_state):
        pass

    def build_obs(self, player, state, previous_action: np.ndarray):
        inverted = player.team_num == common_values.ORANGE_TEAM
        ball = state.inverted_ball if inverted else state.ball
        player_car = player.inverted_car_data if inverted else player.car_data

        obs = [ball.position[:2] * self.POS_COEF,
               ball.linear_velocity[:2] * self.LIN_VEL_COEF,
               player_car.position[:2] * self.POS_COEF,
               player_car.forward()[:2],
               player_car.up()[:2],
               player_car.linear_velocity[:2] * self.LIN_VEL_COEF,
               player_car.angular_velocity[:2] * self.ANG_VEL_COEF
               ]

        return np.concatenate(obs)


if __name__ == '__main__':
    # Make the default rlgym environment
    env = rlgym.make(reward_fn=Reward(),
                     game_speed=100,
                     terminal_conditions=(NoTouchTimeoutCondition(500), GoalScoredCondition()),
                     self_play=False,
                     state_setter=RandomState(),
                     action_parser=CustomActionParser(),
                     obs_builder=CustomObsBuilder())
    # Initialize PPO from stable_baselines3
    policy_kwargs = dict(activation_fn=torch.tanh,
                         net_arch=[dict(pi=[8, 8, 4], vf=[8, 4, 4])])
    model = PPO("MlpPolicy", env=env, verbose=1)

    # Train our agent!
    model.learn(total_timesteps=int(1e7))

    model.save("models/v0_2")
