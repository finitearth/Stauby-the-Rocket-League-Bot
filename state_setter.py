from rlgym.utils.state_setters import RandomState
from numpy import random as rand
from rlgym.utils.math import rand_vec3
from rlgym.utils.state_setters.random_state import YAW_MAX

X_MAX = 7000
Y_MAX = 9000


class StateSetter(RandomState):
    def __init__(self, x_std=1000, y_std=1000):
        super().__init__()
        self.x_std = x_std
        self.y_std = y_std
        self.car_pos = None

    def reset(self, state_wrapper):
        self._reset_cars_random(state_wrapper, self.cars_on_ground, self.cars_rand_speed)
        self._reset_ball_random(state_wrapper, self.ball_rand_speed)

    def _reset_ball_random(self, state_wrapper, random_speed=False):
        state_wrapper.ball.set_pos(
            rand.normal(self.car_pos[0], self.x_std, size=(1,)).clip(-X_MAX/2, +X_MAX/2),# * X_MAX - X_MAX/2,
            rand.normal(self.car_pos[1], self.y_std, size=(1,)).clip(-Y_MAX/2, +Y_MAX/2),
            100
        )

    def _reset_cars_random(self, state_wrapper, on_ground, random_speed):
        """
        Function to set all cars to a random position.

        :param state_wrapper: StateWrapper object to be modified.
        :param on_ground: Boolean indicating whether to place cars only on the ground.
        :param random_speed: Boolean indicating whether to randomize velocity values.
        """
        for car in state_wrapper.cars:
            x = rand.random() * X_MAX - X_MAX/2
            y = rand.random() * Y_MAX - Y_MAX/2
            self.car_pos = x, y
            # z=17 -> on ground
            car.set_pos(x, y, 17)
            car.set_rot(0, rand.random() * YAW_MAX - YAW_MAX/2, 0)

            car.boost = 0
            # car.set_lin_vel(*rand_vec3(2300))
