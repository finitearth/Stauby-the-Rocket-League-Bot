import math

import gym
import numpy as np
from rlgym.utils import ObsBuilder, common_values
from rlgym.utils import math as rlm

from utils import calculate_distance_to_wall

GOAL_POSITION = np.array([0, -4096, 0])


class CustomObsBuilder(ObsBuilder):
    def __init__(self):
        super().__init__()
        self.obs_dim = 10

    def reset(self, initial_state):
        pass

    def get_observation_space(self):
        return gym.spaces.Box(-1, 1, shape=(self.obs_dim,))

    def build_obs(self, player, state, previous_action):
        inverted = player.team_num == common_values.ORANGE_TEAM
        ball = state.inverted_ball if inverted else state.ball
        player_car = player.inverted_car_data if inverted else player.car_data

        obs = [
            rlm.quat_to_euler(player_car.quaternion).sum(),
            player_car.angular_velocity.sum()
        ]
        obs.extend(ball.position[:2])
        obs.extend(ball.linear_velocity[:2])
        obs.extend(player_car.position[:2])
        obs.extend(player_car.linear_velocity[:2])

        return np.array(obs)


def get_polar(diff_vec):
    x = diff_vec[0]
    y = diff_vec[1]
    if x == 0: alpha = 0
    else: alpha = np.arctan(y / x)
    dist = x**2 + y**2
    return dist, alpha


class PolarObsBuilder(CustomObsBuilder):
    def __init__(self):
        super().__init__()
        self.obs_dim = 5

    def build_obs(self, player, state, previous_action):
        inverted = player.team_num == common_values.ORANGE_TEAM
        ball = state.inverted_ball if inverted else state.ball
        player_car = player.inverted_car_data if inverted else player.car_data

        player_angle = rlm.quat_to_euler(player_car.quaternion).sum()
        player_linear_vel = np.sum(player_car.linear_velocity ** 2)

        car_ball_vec = player_car.position - ball.position
        car_ball_dist, car_ball_ang = get_polar(car_ball_vec)
        car_ball_ang = (car_ball_ang - player_angle) % 6.28

        car_goal_vec = player_car.position - GOAL_POSITION
        car_goal_dist, car_goal_ang = get_polar(car_goal_vec)
        car_goal_ang = (car_goal_ang - player_angle) % 6.28

        obs = [
            player_linear_vel,
            car_ball_ang,
            car_ball_dist,
            car_goal_ang,
            car_goal_dist
        ]
        obs = np.asarray(obs)
        mean = np.array([4.56611978e+04, 3.21064016e+00, 4.42694484e+06, 3.08433468e+00, 3.04410152e+07])
        std = np.array([4.03100956e+10, 4.57956887e+00, 1.04389247e+14, 4.11147932e+00, 6.07033630e+14])**(1/2)
        obs -= mean
        obs /= std
        return obs
