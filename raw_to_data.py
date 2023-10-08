import os
import numpy as np
from pandas import read_csv

from file_io import write_to_file, extract_map
from plotting import plot_map


def filter_and_sort(x, y, *args):
    """
    Remove any duplicate coordinates and sort data along x and y.

    :param x: x-coordinates
    :param y: y-coordinates
    :param args: arbitrary number of data corresponding to each point
    :return:
    """
    # Data processing
    # 1. Removing duplicate points by extracting unique (x,y) pairs
    zz = []
    xy = np.dstack((x, y))[0]
    xyp, index = np.unique(xy, axis=0, return_index=True)
    x = x[index]
    y = y[index]
    for arg in args:
        z = arg[index]
        zz.append(z)
    # 2. Creating a regular grid from unique x and y coordinates
    xx, yy, zz = matrix_from_point_data(x, y, zz)
    return xx.ravel(), yy.ravel(), *[z.ravel() for z in zz]


def matrix_from_point_data(x, y, *args):
    """
    Convert point data into matrix data.
    Missing values in args will be substituted with nan.

    :param x: x-coordinates
    :param y: y-coordinates
    :param args: arbitrary number of data corresponding to each point
    :return:
    """
    x_unique = np.unique(x)
    y_unique = np.unique(y)

    # Create index arrays for mapping x and y values to matrix indices
    x_indices = np.searchsorted(x_unique, x)
    y_indices = np.searchsorted(y_unique, y)

    # Create an empty matrix with NaN values
    zz = []
    for arg in args:
        z = np.full((y_unique.size, x_unique.size), np.nan)
        # Assign z values to the corresponding matrix positions
        z[y_indices, x_indices] = arg
        zz.append(z)

    xx, yy = np.meshgrid(x_unique, y_unique)

    return xx, yy, *zz


def plot_graph_from_data(fname, **kwargs):
    data = read_csv(fname, delimiter='\t').to_numpy()
    plot_map(*data.T, kwargs)


if __name__ == '__main__':
    directory = 'sim_data\\exps4'
    output_file = 'r_max_tau_p_o.txt'
    output_file1 = 'R_ind_tau_p_o.txt'
    column_names = ('p_o', 'tau_r', 'r_max')
    column_names1 = ('p_o', 'tau_r', 'R_ind')
    extract = ['p_o', 'tau_r', 'fwhm', 'r_max', 'R_ind']
    output_filepath = os.path.join(directory, output_file)
    output_filepath1 = os.path.join(directory, output_file1)
    p_o1, tau_r1, fwhm, r_max1, R_ind1 = extract_map(directory, *extract)
    r_max_n1 = r_max1 / fwhm
    p_o, tau_r, r_max_n, R_ind = filter_and_sort(p_o1, tau_r1, r_max_n1, R_ind1)
    write_to_file(output_filepath, column_names, p_o, tau_r, r_max_n, digits=5, append=True)
    write_to_file(output_filepath1, column_names1, p_o, tau_r, R_ind, digits=5, append=True)
    # save_map_data(directory, *column_names, output_filepath)
    # plot_graph_from_raw(directory, *column_names)
    plot_graph_from_data(output_filepath, title='Peak position map')
    plot_graph_from_data(output_filepath1, title='Relative indent map')
