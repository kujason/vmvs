

import numpy as np


def box_2d_area(box_2d):
    return (box_2d[2] - box_2d[0]) * (box_2d[3] - box_2d[1])
