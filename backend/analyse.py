"""
Several methods for basic curve analysis
"""

from scipy.signal import argrelextrema as argextr
from scipy.optimize import minimize_scalar, basinhopping
from scipy.interpolate import CubicSpline
import numpy as np

y_global_min = -1


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
    y_max_prelim = y[n]
    if abs(y_max_prelim - y.max()) < 1e-5:
        return 0, y[n]
    ind = (y[n:] > 1e-8).nonzero()[0][-1]  # for reduction of the optimization window
    x_max = x[n + ind]
    init_guess = y.max()
    y_max_ind = np.argmax(y)
    x_max_init = x[y_max_ind]
    n_iters = 1000
    step_size = x.max() / 200
    # res = minimize_scalar(obj_func, bounds=(0, x_max), method='bounded')
    res = basinhopping(obj_func, x_max_init,
                       minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': [(0, x_max)]}, T=5,
                       niter=n_iters, stepsize=step_size,
                       # disp=True,
                       # callback=evaluate_minima
                       )
    # maxima = argextr(y, np.greater)
    # y_max = y[maxima]
    # x_max = x[maxima]
    x_max = np.array(res.x)
    y_max = np.array(-res.fun)

    return x_max, y_max


def evaluate_minima(x, f, accepted):
    """
    Evaluate minima during basinhopping optimization
    :param x: x-coordinates
    :param f: y-coordinates
    :param accepted: bool
    :return: None
    """

    if f < 1e-12:
        return True


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
