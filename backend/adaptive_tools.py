"""
A collection for tools for working with adaptive learners.
"""

from copy import deepcopy
from collections import OrderedDict

import adaptive
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import LinearNDInterpolator

from backend.plotting import plot_map


def learner_load_full(fname, compress=True, type='2D'):
    """
    Load a learner from a file with bonds and scale.
    :param fname: filename
    :param compress: True for loading with compression
    :param type: '2D' or 'ND'
    :return: Learner2D
    """
    # Because rebounding the learner to a larger bounds is not working, we have to set the bounds manually
    # First, we get the bonds from the data
    learner = adaptive.Learner2D(min, ((0, 1), (0, 1)))
    learner.load(fname, compress=compress)
    data = learner_data_to_numpy(learner)
    bounds = bounds_2d_from_data(data)
    # Then we create a new learner with the correct bounds
    learner = adaptive.Learner2D(min, bounds)
    # And load the data
    learner.load(fname, compress=compress)

    return learner


def learner_interpolator(learner: adaptive.Learner2D):
    """
    Return the interpolator of the learner.
    :param learner: Learner2D, LearnerND
    :return: LinearNDInterpolator
    """
    # The learner uses Scipy's linear interpolator, so we can get it by just creating one with the learner's data.
    data = learner.to_numpy().T
    points = data[:-1].T
    values = data[-1]
    interpolator = LinearNDInterpolator(points, values, rescale=True)
    return interpolator


def plot_learner(learner: adaptive.Learner2D, n_points=300, **kwargs):
    """
    Plot the learner as a surface map using matplotlib.
    :param n_points: number of points along x and y-axis
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
    learner_new = adaptive.Learner2D(min, bounds)  # min is a just a stub function
    learner_new.copy_from(learner)
    learner_new.function = learner.function
    return learner_new


def combine_learners(learner1: adaptive.Learner2D, learner2: adaptive.Learner2D):
    """
    Combine two learners into one.
    Allows several routines like stitching learners with adjacent bounds, increasing triangulation on a specific area,
    sequentially extend the learner with new data, etc.
    :param learner1: Learner2D
    :param learner2: Learner2D
    :return: Learner2D with combined data and bounds
    """
    bounds = get_combined_bounds(learner1.bounds, learner2.bounds)
    learner = adaptive.Learner2D(min, bounds)  # min is a just a stub function
    data1 = learner1.data
    data2 = learner2.data
    data = deepcopy(data1)
    data.update(data2)
    learner.data = data
    return learner


def stack_learners(learner1: adaptive.Learner2D, learner2: adaptive.Learner2D, val1, val2):
    """
    Stack two learners along a new dimension.
    If two or more learners are mapping the same function with a parameter that is fixed for a single learner,
    but varies between these learners, it is possible to stack them along this parameter.
    :param learner1: Learner2D
    :param learner2: Learner2D
    :return: LearnerND with combined data and bounds
    """
    data1 = learner1.data
    data1_nd = OrderedDict({(key[0], key[1], val1): value for key, value in data1.items()})
    data2 = learner2.data
    data2_nd = OrderedDict({(key[0], key[1], val2): value for key, value in data2.items()})
    data = deepcopy(data1_nd)
    data.update(data2_nd)
    bounds = get_combined_bounds(learner1.bounds, learner2.bounds)
    bounds_nd = (bounds[0], bounds[1], (min(val1, val2), max(val1, val2)))
    learner = adaptive.LearnerND(min, bounds_nd)
    learner.data = data
    return learner


def convert_Learner2D_to_LearnerND(learner: adaptive.Learner2D, ):
    """
    Convert a 2D learner to an N-dimensional learner.
    :param learner: Learner2D
    :return: LearnerND
    """
    bounds = learner.bounds
    learner_nd = adaptive.LearnerND(min, bounds)
    data = learner.data
    data_nd = {key: np.array([np.append(point, 0) for point in value]) for key, value in data.items()}
    learner_nd.data = data_nd
    return learner_nd


def get_combined_bounds(*bounds_list):
    """
    Calculate combined bounds from several input bounds.

    :param bounds_list: A list of bounds, where each bound is a tuple of tuples [(x_min, x_max), (y_min, y_max)]
    :return: A tuple representing the combined bounds [(x_min, x_max), (y_min, y_max)]
    """
    x_min = min(bounds[0][0] for bounds in bounds_list)
    x_max = max(bounds[0][1] for bounds in bounds_list)
    y_min = min(bounds[1][0] for bounds in bounds_list)
    y_max = max(bounds[1][1] for bounds in bounds_list)
    bounds = ((x_min, x_max), (y_min, y_max))
    return bounds


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


def bounds_2d_from_data(data):
    """
    Calculate 2D bounds for a learner from data.
    :param data: 2D numpy array
    :return: list of tuples [(x_min, x_max), (y_min, y_max)]
    """
    x, y = data[:, 0], data[:, 1]
    return ((x.min(), x.max()), (y.min(), y.max()))


def learner_to_txt(learner, fname):
    """
    Save learner's data to a text file.
    :param learner: Learner2D
    :param fname: filename
    :return: None
    """
    data = learner.to_numpy()
    n_cols = data.shape[1]
    header = '\t'.join([f'col_{i+1}' for i in range(n_cols)])
    np.savetxt(fname, data, header=header, delimiter='\t')


def learner_data_to_numpy(learner):
    """
    Convert learner's data to a numpy array.
    :param learner: Learner2D or LearnerND
    :return: 2D numpy array
    """
    data = learner.data
    if type(data) == dict:
        data = data['data']
    arr_list = [[*key, value] for key, value in data.items()]
    arr = np.asarray(arr_list)
    return arr


if __name__ == '__main__':
    pass
    # Example of using the functions
    fname = r'/home/alex/PycharmProjects/febid_2d/data/maps/R_ind_interp_2.0.int'
    fname1 = r'/home/alex/PycharmProjects/febid_2d/data/maps/R_ind_interp_3.0.int'
    learner = learner_load_full(fname)
    arr = learner_data_to_numpy(learner)
    # plot_learner(learner)
    learner_rebound(learner, ((0, 2), (1, 1000)))
    fname_save = fname.replace('.int', '.txt')
    learner_to_txt(learner, fname_save)
    bounds = bounds_2d_from_data(learner.to_numpy())
    print(bounds)
    learner_interpolator(learner)
    learner1 = learner_load_full(fname1)
    learner_nd = stack_learners(learner, learner1, 2, 3)
    # combine_learners(learner1=learner, learner2=learner)
    grid1 = generate_lin_grid((0, 1), (0, 1), 10)
    plt.scatter(grid1[0][0], np.full_like(grid1[0][0], 1))
    plt.show()
    grid2 = generate_log_grid((1e-3, 1), (1e-3, 1), 10)
    plt.scatter(grid2[0][0], np.full_like(grid2[0][0], 1))
    plt.show()