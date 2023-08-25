"""
A dummy interface function for obtaining a beam profile.
"""

import numpy as np


def generate_profile(r, type='gauss', param1=2.5, param2=1):
    """
    Generate a beam profile based on the provided grid.

    Three profile types are available: 'gauss', 'super_gauss' and 'top_hat'
    'gauss' generates a classic gaussian curve.
    'super_gauss' generates a higher-order gaussian with abrupt edges,
    'top_hat' generates a flat-top profile with adjustable edge profile.

    For 'gauss' param1 is standard deviation, param2 is ignored.
    For 'super_gauss' param1 is the same, param2 is order (classic gaussian has order of 1).
    For 'top_hat' param1 is beam radius, param2 is edge abruptness.

    :param r: grid
    :param type: type of generated profile
    :param param1: standard deviation or beam radius, see descr.
    :param param2: order/edge abruptness
    :return:
    """
    if type == 'gauss':
        return get_gauss(r, param1, 1)
    elif type == 'super_gauss':
        return get_gauss(r, param1, param2)
    elif type == 'top_hat':
        return get_top_hat(r, param1, param2)


def get_gauss(r, st_dev, n=1):
    """
    Generate gaussian distribution.

    :param r: grid
    :param a: standard deviation
    :param n: order of the Gauss-function
    :return:
    """
    return np.exp(-(r ** 2 / (2 * st_dev ** 2)) ** n)


def get_top_hat(r, h, b):
    """
    Generate a top-hat profile.

    :param r: grid
    :param h: beam radius
    :param b: edge abruptness
    :return:
    """
    return 1 / np.exp(b * (r / h - 1))

