import numpy as np

X_MAX = 7000.
Y_MAX = 9000.


def calculate_distance_to_wall(pos):
    x = pos[0]
    y = pos[1]
    min_x = min(abs(x-X_MAX), abs(y+Y_MAX))
    min_y = min(abs(y-Y_MAX), abs(y+Y_MAX))

    return min(min_x, min_y)
