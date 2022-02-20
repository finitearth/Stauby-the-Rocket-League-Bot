import math
import numpy as np
from rlgym.utils import ObsBuilder, common_values


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


class CustomActionParser:
    def __init__(self):
        super().__init__()

    def parse_actions(self, actions):
        filled_action = np.zeros(8)
        filled_action[[0, 1, 6]] = (actions[0]+1)/2, actions[1], actions[2]>0

        return filled_action
