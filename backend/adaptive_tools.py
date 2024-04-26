import adaptive
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from backend.plotting import plot_map


def learner_load_full(fname, compress=True):
    """
    Load a 2D learner from a file with bonds and xy scale.
    :param fname: filename
    :param compress: True for loading with compression
    :return: Learner2D
    """
    learner = adaptive.Learner2D(min, [(0, 1), (0, 1)])
    learner.load(fname, compress=compress)
    data = learner.to_numpy().T
    x, y, z = data
    learner = adaptive.Learner2D(min, [(x.min(), x.max()), (y.min(), y.max())])
    learner.load(fname, compress=compress)

    return learner


def learner_interpolator(learner: adaptive.Learner2D):
    """
    Return the interpolator of the learner.
    :param learner: Learner2D
    :return: LinearNDInterpolator
    """
    data = learner.to_numpy()
    interpolator = LinearNDInterpolator(data[:, 0:2], data[:, 2], rescale=True)
    return interpolator


def plot_learner(learner: adaptive.Learner2D, n_points=300, **kwargs):
    """
    Plot the learner as a surface map using matplotlib.
    :param n_points: number of points along x and y axis
    :param learner: Learner2D
    :param kwargs: kwargs for backend.plotting.plot_map()
    :return: None
    """
    data_grid = learner.interpolated_on_grid(n_points)
    xx, yy, zz = data_grid
    zz = zz.T
    fig, ax = plot_map(xx, yy, zz, **kwargs)
    return fig, ax


def learner_rebound(learner: adaptive.Learner2D, bounds):
    """
    Rebound the learner to new bounds.
    :param learner: Learner2D
    :param bounds: list of tuples [(x_min, x_max), (y_min, y_max)]
    :return: None
    """
    learner_new = adaptive.Learner2D(min, bounds)
    learner_new.copy_from(learner)
    learner_new.function = learner.function
    return learner_new


def generate_lin_grid(x_bounds, y_bounds, num_points):
    """
    Generate a 2D linear grid of points within specified bounds.

    :param x_bounds: A tuple specifying the lower and upper bounds for x.
    :param y_bounds: A tuple specifying the lower and upper bounds for y.
    :param num_points: The number of points to be generated along both dimensions.
    :return: A tuple of 2D arrays representing the x and y coordinates of the points in the grid.
    """
    x = np.linspace(x_bounds[0], x_bounds[1], num_points)
    y = np.linspace(y_bounds[0], y_bounds[1], num_points)
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def generate_log_grid(x_bounds, y_bounds, num_points):
    """
    Generate a 2D grid of points within specified bounds with logarithmic transformation.

    :param x_bounds: A tuple specifying the lower and upper bounds for x.
    :param y_bounds: A tuple specifying the lower and upper bounds for y.
    :param num_points: The number of points to be generated along both dimensions.
    :return: A tuple of 2D arrays representing the x and y coordinates of the points in the grid.
    """
    x = np.logspace(np.log10(x_bounds[0]), np.log10(x_bounds[1]), num_points)
    y = np.logspace(np.log10(y_bounds[0]), np.log10(y_bounds[1]), num_points)
    xx, yy = np.meshgrid(x, y)
    return xx, yy
