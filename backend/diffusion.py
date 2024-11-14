import numpy as np

def laplace_1d(n):
    """
    Apply 1D Laplace operator to the array.
    :param n: 1D array
    :return: transformed array
    """
    n_out = np.copy(n)
    n_out[:] *= -2
    n_out[:-1] += n[1:]
    n_out[-1] += n[-1]

    n_out[1:] += n[:-1]
    n_out[0] += n[0]
    return n_out