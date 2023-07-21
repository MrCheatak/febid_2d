"""
Several methods for basic curve analysis
"""


from scipy.signal import argrelextrema as argextr
import numpy as np


def convergence_analysis(y1, y2):
    """
    Calculate residual sum of squares between two curves.
    Curves must share the same x-coordinates.
    :param y1: curve 1
    :param y2: curve 2
    :return: float
    """
    # residual sum of squares
    ss_res = np.sum((y1 - y2) ** 2)

    # total sum of squares
    ss_tot = np.sum((y1 - np.mean(1)) ** 2)

    # r-squared
    r2 = 1 - (ss_res / ss_tot)

    return r2


def get_peak(x, y):
    """
    Get all peaks of a curve
    :param x: x-coordinates
    :param y: y-coordinates
    :return: [x positions], [y positions]
    """
    maxima = argextr(y, np.greater)
    y_max = y[maxima]
    x_max = x[maxima]

    return x_max, y_max


def deposit_fwhm(x, y):
    """
    Get FWHM of a curve. Curve must have bell curve-like behaviour.
    :param x: x-coordinates
    :param y: y-coordinates
    :return: float
    """
    hm_d1 = (y.max() - y.min()) / 2
    delta = y - hm_d1
    cell = np.abs(delta).min()
    item1 = (np.abs(delta) == cell).nonzero()[0]
    d1 = x[item1]
    fwhm = np.fabs(2 * d1)[0]
    return fwhm
