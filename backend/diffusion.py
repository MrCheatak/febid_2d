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


class CrankNicolsonRadialSolver:
    """
    Crank-Nicolson solver for radial diffusion equation.

    Solves: ∂n/∂t = D * (∂²n/∂r² + (1/r)∂n/∂r) + S(r,t)

    Uses implicit Crank-Nicolson scheme with LU factorization of the
    coefficient matrix for efficient time-stepping.

    The matrix is factored once during initialization and reused for all steps.
    """

    def __init__(self, r, D, dt, bc_outer="neumann", dirichlet_value=0):
        """
        Initialize the Crank-Nicolson solver.

        :param r: 1D array of radial positions (including r=0)
        :param D: diffusion coefficient
        :param dt: time step
        :param bc_outer: "neumann" (zero gradient) or "dirichlet"
        :param dirichlet_value: value for Dirichlet BC at outer boundary
        """
        self.r = r
        self.N = len(r)
        self.D = D
        self.dt = dt
        self.dr = r[1] - r[0]  # assuming uniform grid
        self.bc_outer = bc_outer
        self.dirichlet_value = dirichlet_value

        # precompute common scalars
        self._alpha = 0.5 * self.dt * self.D
        self._inv_dr2 = 1.0 / (self.dr * self.dr)

        # precompute coefficients for interior points (i = 1..N-2)
        # coef_2nd is constant for all interior points; coef_1st depends on r[i]
        if self.N > 2:
            interior_r = self.r[1:-1]
            self._coef_2nd_interior = self._alpha * self._inv_dr2
            self._coef_1st_interior = self._alpha / (2.0 * self.dr * interior_r)
        else:
            self._coef_2nd_interior = None
            self._coef_1st_interior = None

        # Build the tridiagonal matrix for implicit step in banded form
        # (I - 0.5*dt*D*L) * n^{k+1} = (I + 0.5*dt*D*L) * n^k + dt*S
        self.ab = self._build_implicit_banded_matrix()

    def _build_implicit_banded_matrix(self):
        """Build the coefficient matrix for the implicit step in banded form.

        Matrix form: (I - 0.5*dt*D*L) where L is the radial Laplacian operator.

        The returned array `ab` has shape (3, N) and SciPy `solve_banded`
        layout with one upper and one lower diagonal: `ab[0]` is the
        upper diagonal (shifted right by one), `ab[1]` the main diagonal,
        and `ab[2]` the lower diagonal (shifted left by one).
        """
        N = self.N
        dr = self.dr
        r = self.r
        alpha = self._alpha

        lower = np.zeros(N)
        main = np.ones(N)
        upper = np.zeros(N)

        # Center point (r=0): use symmetry boundary condition
        # Laplacian at r=0: 4*(n[1] - n[0])/dr^2
        main[0] = 1.0 + 4.0 * alpha * self._inv_dr2
        upper[0] = -4.0 * alpha * self._inv_dr2

        # Interior points (i = 1 to N-2)
        if N > 2:
            coef_2nd = alpha * self._inv_dr2
            coef_1st = alpha / (2.0 * dr * r[1:-1])

            lower[1:-1] = -(coef_2nd - coef_1st)
            main[1:-1] = 1.0 + 2.0 * coef_2nd
            upper[1:-1] = -(coef_2nd + coef_1st)

        # Outer boundary (i = N-1)
        if self.bc_outer == "neumann":
            # Zero gradient: n/r = 0
            # Using ghost point: n_N = n_{N-2}
            # Laplacian: 2*(n_{N-2} - n_{N-1})/dr^2
            lower[-1] = -2.0 * alpha * self._inv_dr2
            main[-1] = 1.0 + 2.0 * alpha * self._inv_dr2
        elif self.bc_outer == "dirichlet":
            # Fixed value at boundary
            # Ghost point: n_N = 2*n_bc - n_{N-1}
            ri = r[-1]
            coef_2nd = alpha * self._inv_dr2
            coef_1st = alpha / (2.0 * dr * ri)

            lower[-1] = -(coef_2nd - coef_1st)
            main[-1] = 1.0 + 2.0 * coef_2nd + (coef_2nd + coef_1st)

        # Convert to banded form for solve_banded with (1, 1)
        ab = np.zeros((3, N))
        ab[0, 1:] = upper[:-1]
        ab[1, :] = main
        ab[2, :-1] = lower[1:]
        return ab

    def _apply_explicit_operator(self, n, source=None):
        """
        Apply the explicit part: (I + 0.5*dt*D*L) * n + 0.5*dt*S
        where L is the radial Laplacian.

        :param n: concentration array at current time
        :param source: source term array (optional)
        :return: right-hand side for the implicit solve
        """
        N = self.N
        dr = self.dr
        r = self.r
        alpha = self._alpha

        rhs = np.copy(n)

        # Center point (r=0)
        rhs[0] = n[0] + 4.0 * alpha * (n[1] - n[0]) * self._inv_dr2

        # Interior points (vectorized if there are any)
        if N > 2:
            ip = n[2:]
            i0 = n[1:-1]
            im = n[:-2]

            coef_2nd = alpha * self._inv_dr2
            coef_1st = alpha / (2.0 * dr * r[1:-1])

            lap = (ip - 2.0 * i0 + im) * coef_2nd
            lap += (ip - im) * coef_1st
            rhs[1:-1] = i0 + lap

        # Outer boundary
        if self.bc_outer == "neumann":
            rhs[-1] = n[-1] + 2.0 * alpha * (n[-2] - n[-1]) * self._inv_dr2
        elif self.bc_outer == "dirichlet":
            ri = r[-1]
            coef_2nd = alpha * self._inv_dr2
            coef_1st = alpha / (2.0 * dr * ri)
            # Ghost point contribution
            n_ghost = 2.0 * self.dirichlet_value - n[-1]
            lap = (n_ghost - 2.0 * n[-1] + n[-2]) * coef_2nd
            lap += (n_ghost - n[-2]) * coef_1st
            rhs[-1] = n[-1] + lap

        # Add source term if provided
        if source is not None:
            rhs += 0.5 * self.dt * source

        return rhs

    def step(self, n, source=None):
        """
        Advance one time step using Crank-Nicolson scheme.

        :param n: concentration array at current time
        :param source: source term array (optional), evaluated at current time
        :return: concentration array at next time step
        """
        # Compute right-hand side
        rhs = self._apply_explicit_operator(n, source)

        # Solve linear system using banded solver; A is constant in time
        n_new = solve_banded((1, 1), self.ab, rhs)

        return n_new

    def solve(self, n0, t_final, source_func=None, store_every=1):
        """
        Solve the diffusion equation from t=0 to t=t_final.

        :param n0: initial concentration array
        :param t_final: final time
        :param source_func: function(r, t) that returns source term array (optional)
        :param store_every: store solution every N steps (default: 1)
        :return: (times, solutions) where solutions is a 2D array [time_idx, space_idx]
        """
        n_steps = int(np.ceil(t_final / self.dt))
        n = np.copy(n0)

        # Storage
        n_stored = max(1, n_steps // store_every + 1)
        solutions = np.zeros((n_stored, self.N))
        times = np.zeros(n_stored)

        solutions[0, :] = n
        times[0] = 0.0

        store_idx = 1
        for step in range(n_steps):
            t = step * self.dt

            # Evaluate source if provided
            source = None
            if source_func is not None:
                source = source_func(self.r, t)

            # Take a step
            n = self.step(n, source)

            # Store if needed
            if (step + 1) % store_every == 0 and store_idx < n_stored:
                solutions[store_idx, :] = n
                times[store_idx] = (step + 1) * self.dt
                store_idx += 1

        # Ensure final state is stored
        if times[-1] < t_final - 1e-10:
            solutions[-1, :] = n
            times[-1] = n_steps * self.dt

        return times, solutions
