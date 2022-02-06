from rlgym.utils.state_setters import RandomState
from numpy import random as rand


X_MAX = 7000
Y_MAX = 9000


class StateSetter(RandomState):
    def __init__(self):
        super().__init__(ball_rand_speed=False, cars_rand_speed=False, cars_on_ground=True)

    def _reset_ball_random(self, state_wrapper, random_speed=False):
        state_wrapper.ball.set_pos(
            rand.random() * X_MAX - X_MAX/2,
            rand.random() * Y_MAX - Y_MAX/2,
            100
        )
