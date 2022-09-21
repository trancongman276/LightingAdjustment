import math

import numpy as np


def HVS_func(X, a, f, m, x_glob_a, x_local_a, V_max=255):
    """
    **Implementation of Human Vision System**\n
    The parameter a controls the contribution of global adaptation level (*x_glob_a*) and local adaptation level (*x_local_a*)\n
    The image contrast is enhanced when *m* increased\n
    The image brightness is enhanced when *f* is decreasing

    :param X:           Input image
    :param a:           Parameter a (for adaptation level)
    :param f:           Parameter f (for brightness)
    :param m:           Parameter m (for contrast)
    :param x_glob_a:    Global adaptation level
    :param x_local_a:   Local adaptation level
    :param V_max:       Maximum range of output value
    :return:            Enhanced image
    """
    assert 0 <= a <= 1, "a must be in range [0, 1]"
    assert 0 <= m < 3.1, "m must be in range [0, 3.1)"
    assert math.exp(-8) <= f <= math.exp(8), "f must be in range [exp(-8), exp(8)]"

    # Calculate the adaptation level
    x_a = a * x_local_a + (1 - a) * x_glob_a

    # Calculate the semi-saturation constant
    ssc = (f * x_a) ** m
    rs = (X / (X + ssc + 1e-9)) * V_max
    return rs


def entropy_H(X):
    """
    Calculate the pixels probability in given image
    :param X: Input image
    :return:  Probability
    """
    H = 0
    w, h, c = X.shape
    total_pixel = w * h * c
    values, counts = np.unique(X, return_counts=True)
    for c in counts:
        p = c / total_pixel
        H += p * math.log2(p)
    H = -H
    return H
