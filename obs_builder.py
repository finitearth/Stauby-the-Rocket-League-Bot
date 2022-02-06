import math

import gym
import numpy as np
from rlgym.utils import ObsBuilder, common_values
from rlgym.utils import math as rlm

GOAL_POSITION = np.array([0, 4096, 0])


class CustomObsBuilder(ObsBuilder):
    def __init__(self, pos_coef=1 / 2300, ang_coef=1 / math.pi, lin_vel_coef=1 / 2300, ang_vel_coef=1 / math.pi):
        super().__init__()
        self.POS_COEF = pos_coef
        self.ANG_COEF = ang_coef
        self.LIN_VEL_COEF = lin_vel_coef
        self.ANG_VEL_COEF = ang_vel_coef

    def reset(self, initial_state):
        pass

    def get_observation_space(self):
        return gym.spaces.Box(-1, 1, shape=(14,))

    def build_obs(self, player, state, previous_action):
        inverted = player.team_num == common_values.ORANGE_TEAM
        ball = state.inverted_ball if inverted else state.ball
        player_car = player.inverted_car_data if inverted else player.car_data

        obs = [
            ball.position[:2] * self.POS_COEF,
            ball.linear_velocity[:2] * self.LIN_VEL_COEF,
            player_car.position[:2] * self.POS_COEF,
            player_car.forward()[:2],
            player_car.up()[:2],
            player_car.linear_velocity[:2] * self.LIN_VEL_COEF,
            player_car.angular_velocity[:2] * self.ANG_VEL_COEF
        ]

        return np.concatenate(obs)


def get_polar(diff_vec):
    x = diff_vec[0]
    y = diff_vec[1]
    if x == 0: alpha = 0
    else: alpha = np.arctan(y / x)
    dist = (x**2 + y**2)**(1/2)
    return dist, alpha


class PolarObsBuilder(CustomObsBuilder):
    def __init__(self):
        super().__init__()

    def get_observation_space(self):
        return gym.spaces.Box(-10, +10, shape=(6,))

    def build_obs(self, player, state, previous_action):
        inverted = False#player.team_num == common_values.ORANGE_TEAM
        ball = state.inverted_ball if inverted else state.ball
        player_car = player.inverted_car_data if inverted else player.car_data

        player_angle = rlm.quat_to_euler(player_car.quaternion)[1]
        player_linear_vel = np.sqrt(np.sum(player_car.linear_velocity ** 2))
        player_ang_vel = player_car.angular_velocity[1]

        ball_vel = np.sqrt(np.sum(ball.linear_velocity ** 2))

        car_ball_vec = player_car.position - ball.position
        car_ball_dist, car_ball_ang = get_polar(car_ball_vec)
        car_ball_ang -= player_angle

        car_goal_vec = player_car.position - GOAL_POSITION
        car_goal_dist, car_goal_ang = get_polar(car_goal_vec)
        car_goal_ang -= player_angle

        obs = [
            player_linear_vel * self.LIN_VEL_COEF,
            player_ang_vel * self.ANG_VEL_COEF,
            # player_angle * self.ANG_COEF,
            # ball_vel * self.LIN_VEL_COEF,
            car_ball_ang * self.ANG_COEF,
            car_ball_dist * self.POS_COEF,
            car_goal_ang * self.ANG_COEF,
            car_goal_dist * self.POS_COEF
        ]
        return np.asarray(obs)
