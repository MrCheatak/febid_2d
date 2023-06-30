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


def find_crossing(a, b, a_ref, b_ref):
    """
    a and b are 2D maps.

    :param a:
    :param b:
    :param a_val:
    :param b_val:
    :return:
    """

    inds = []
    eps = 0.1
    while True:
        ind1 = np.fabs(a[:, 2] - a_ref) <= eps * a_ref
        ind2 = np.fabs(b[:, 2] - b_ref) <= eps * b_ref
        x1 = r_max[ind1, 0]
        y1 = r_max[ind1, 1]
        x2 = R_ind[ind2, 0]
        y2 = R_ind[ind2, 1]
        # Finding intersection along X
        interx = np.intersect1d(x1, x2, return_indices=True)
        x11 = x1[interx[1]]
        # x22 = x2[interx[2]]
        y11 = y1[interx[1]]
        y22 = y2[interx[2]]
        # Finding intersection along Y
        intery = np.intersect1d(y11, y22, return_indices=True)
        inds = intery[1]
        # x = x11[intery[1]]
        # y = intery[0]
        if len(inds) > 1:
            x = x11[intery[1]]
            y = intery[0]
            eps /= 2
        else:
            break
    return x, y


if __name__ == '__main__':
    r_max_file = 'tau_p_7.txt'
    R_ind_file = 'sim_data\\R_ind_tau_p_o.txt'

    r_max_ref = 0.9
    R_ind_ref = 0.2

    r_max = np.genfromtxt(r_max_file, delimiter='\t', skip_header=1)
    R_ind = np.genfromtxt(R_ind_file, delimiter='\t', skip_header=1)
    r_max, R_ind = trim(r_max, R_ind)

    x, y = find_crossing(r_max, R_ind, r_max_ref, R_ind_ref)

    a = 0

