import os
import numpy as np



def trim(a, b):
    """
    Discard all the points that lie outside a common value interval of the two arrays.
    (Trim value ranges of the two arrays)
    :param a:
    :param b:
    :return:
    """
    # Trimming along X
    if a[:, 0].max() > b[:, 0].max():
        a = a[a[:, 0] < b[:, 0].max()]
    else:
        b = b[b[:, 0] < b[:, 0].max()]
    # Trimming along Y
    if a[:, 1].max() > b[:, 1].max():
        a = a[a[:, 1] < b[:, 1].max()]
    else:
        b = b[b[:, 1] < a[:, 1].max()]

    return a, b

def intersect2D(a, b):
  """
  Find row intersection between 2D numpy arrays, a and b.
  Returns another numpy array with shared rows
  """
  return np.array([x for x in set(tuple(x) for x in a) & set(tuple(x) for x in b)])

def find_crossing(a, b, a_ref, b_ref):
    """
    Find point on the map, where regions on maps a and b are taking specified values.
    Maps must be resolved on the same x-y grid share a common region.

    More than one point may be returned.

    :param a: map of the first variable
    :param b: map of the second variable
    :param a_ref: value of a on the map
    :param b_ref: value of b on the map
    :return: tuple(x coords), tuple(y coords)
    """

    # Algorithm works step-wise by narrowing value window (increasing accuracy) on maps.
    # A common region is found by finding intersections along axes.
    x, y = (), ()
    eps = 0.1
    while True:
        ind1 = np.fabs(a[:, 2] - a_ref) <= eps * a_ref
        ind2 = np.fabs(b[:, 2] - b_ref) <= eps * b_ref
        a_temp = a[ind1, 0:2]
        b_temp = b[ind2, 0:2]
        inter = intersect2D(a_temp, b_temp)
        if inter.shape[0] > 0:
            x, y = inter[:, 0], inter[:, 1]
            eps /= 2
        else:
            x = x.mean()
            y = y.mean()
            break
    return x, y


if __name__ == '__main__':
    r_max_file = 'sim_data\\exps2\\r_max_tau_p_o.txt'
    R_ind_file = 'sim_data\\exps2\\R_ind_tau_p_o.txt'

    r_max_ref = 0.89
    R_ind_ref = 0.379

    r_max = np.genfromtxt(r_max_file, delimiter='\t', skip_header=1)
    R_ind = np.genfromtxt(R_ind_file, delimiter='\t', skip_header=1)
    r_max, R_ind = trim(r_max, R_ind)

    x, y = find_crossing(r_max, R_ind, r_max_ref, R_ind_ref)
    print(x, y)
    a = 0

