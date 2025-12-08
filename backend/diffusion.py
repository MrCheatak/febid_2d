import numpy as np
from scipy.linalg import solve_banded

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


def laplacian_radial_1d(n, r, dr, bc_outer="neumann", dirichlet_value=0):
    """
    Apply radial Laplace operator to the array.

    Axisymmetric (cylindrical) radial Laplacian for n(r), r in [0, R].
    Returns (∂²n/∂r² + (1/r)∂n/∂r) at the grid points.
    - bc_outer: "neumann" (zero gradient) or "dirichlet"
    :param n: 1D concentration array
    :param r: radial positions
    :param dr: grid spacing
    :return: transformed array
    """
    n_out = np.copy(n)
    # n_out[:] *= -2  # actually uselss, since the values are overwritten
    # interior: i = 1..N-2
    ip = n[2:]
    i0 = n[1:-1]
    im = n[:-2]
    n_out[1:-1] = (ip - 2 * i0 + im) / dr**2
    n_out[1:-1] += (ip - im) / (2 * dr * r[1:-1])
    # At the center r=0, use symmetry
    n_out[0] = (n[1] - n[0]) * 4 / dr**2

    # At the outer boundary, use zero-flux (Neumann) BC# outer boundary: i = N-1
    if bc_outer == "neumann":
        # ghost n_N = n_{N-2}  -> second-deriv: (n_{N-2} - n_{N-1})/dr^2
        n_out[-1] = 2 * (n[-2] - n[-1]) / dr**2
        # first-deriv term ~ (n_N - n_{N-2})/(2dr r_{N-1}) = 0
    elif bc_outer == "dirichlet":
        # ghost n_N = 2*n(R) - n_{N-1}
        N = len(n)
        nR = dirichlet_value
        nN = 2*nR - n[-1]
        n_out[-1]  = (nN - 2*n[-1] + n[-2]) / (dr*dr)
        n_out[-1] += (nN - n[-2]) / (2*dr) / ( (N-1)*dr )
    else:
        raise ValueError("bc_outer must be 'neumann' or 'dirichlet'.")

    return n_out
