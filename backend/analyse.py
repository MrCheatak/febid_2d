"""
Several methods for basic curve analysis
"""

from scipy.signal import argrelextrema as argextr
from scipy.optimize import minimize_scalar
from scipy.interpolate import CubicSpline
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


def get_peak(x, y, sp=None):
    """
    Get all peaks of a curve.
    :param x: x-coordinates
    :param y: y-coordinates
    :param sp: interpolation function
    :return: x position, y position
    """
    if sp is None:
        sp = CubicSpline(x, y)

    def obj_func(f):
        return -sp(f)

    n = x.size // 2  # center
    ind = (y[n:] > 1e-8).nonzero()[0][-1]  # for reduction of the optimization window
    x_max = x[n + ind]
    res = minimize_scalar(obj_func, bounds=(0, x_max), method='bounded')
    # maxima = argextr(y, np.greater)
    # y_max = y[maxima]
    # x_max = x[maxima]
    x_max = res.x
    y_max = -res.fun

    return x_max, y_max


def deposit_fwhm_legacy1(x, y):
    """
    Get FWHM of a curve. Curve must have bell curve-like behaviour.
    :param x: x-coordinates
    :param y: y-coordinates
    :return: float
    """
    hm_d1 = (y.max() - y.min()) / 2
    # sp = CubicSpline(y, x)
    # fwhm = sp(hm_d1)
    delta = y - hm_d1
    cell = np.abs(delta).min()
    item1 = (np.abs(delta) == cell).nonzero()[0]
    d1 = x[item1]
    fwhm = np.fabs(2 * d1)[0]
    return fwhm


def deposit_fwhm(x, y):
    """
    Get FWHM of a curve. Curve must have bell curve-like behaviour.
    :param x: x-coordinates
    :param y: y-coordinates
    :return: float
    """
    half_height = (y.max() - y.min()) / 2
    d = np.sign(half_height - (y[0:-1])) - np.sign(half_height - (y[1:]))
    left = (d > 0).nonzero()[0][0]
    right = (d < 0).nonzero()[0][-1]
    fwhm = x[right] - x[left]
    return fwhm
