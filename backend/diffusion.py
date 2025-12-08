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
    Crank-Nicolson solver for radial reaction-diffusion equation.

    Uses implicit Crank-Nicolson scheme with LU factorization of the
    coefficient matrix for efficient time-stepping.

    The matrix is factored once during initialization and reused for all steps.

    Solves: ∂n/∂t = D * (∂²n/∂r² + (1/r)∂n/∂r) - K(r)*n + S(r)
    """

    def __init__(self, r, D, dt, reaction_k=0.0, reaction_s=0.0, bc_outer="neumann", dirichlet_value=0):
        """
        :param r: Radial grid
        :param D: Diffusion coefficient
        :param dt: Initial time step
        :param reaction_k: Linear decay rate (scalar or array matching r). Corresponds to (s*F/n0 + 1/tau + sigma*f)
        :param reaction_s: Constant source term (scalar or array). Corresponds to (s*F)
        """
        self.r = r
        self.N = len(r)
        self.D = D
        self.dr = r[1] - r[0]
        self.bc_outer = bc_outer
        self.dirichlet_value = dirichlet_value

        # Store reaction terms
        self.k = reaction_k
        self.s = reaction_s

        self._inv_dr2 = 1.0 / (self.dr * self.dr)

        # Initialize time-dependent parameters
        self.set_dt(dt)

    def set_dt(self, dt):
        """Update time step and rebuild matrix if dt changes."""
        self.dt = dt
        self._alpha = 0.5 * self.dt * self.D

        # Precompute diffusion coefficients based on new alpha
        if self.N > 2:
            interior_r = self.r[1:-1]
            self._coef_2nd_interior = self._alpha * self._inv_dr2
            self._coef_1st_interior = self._alpha / (2.0 * self.dr * interior_r)

        # Rebuild the matrix with new dt
        self.ab = self._build_implicit_banded_matrix()

    def _build_implicit_banded_matrix(self):
        """
        Build (I - dt/2 * (D*Laplacian - K)).
        Result is banded matrix for (I - 0.5*dt*D*L + 0.5*dt*K) * n^{k+1}
        """
        N = self.N
        dr = self.dr
        r = self.r
        alpha = self._alpha

        lower = np.zeros(N)
        main = np.ones(N)
        upper = np.zeros(N)

        # 1. Diffusion Operator Contribution

        # Center point (r=0) symmetry: Laplacian = 4*(n[1]-n[0])/dr^2
        main[0] += 4.0 * alpha * self._inv_dr2
        upper[0] -= 4.0 * alpha * self._inv_dr2

        # Interior points
        if N > 2:
            coef_2nd = self._coef_2nd_interior
            coef_1st = self._coef_1st_interior

            lower[1:-1] -= (coef_2nd - coef_1st)
            main[1:-1] += 2.0 * coef_2nd
            upper[1:-1] -= (coef_2nd + coef_1st)

        # Outer boundary
        if self.bc_outer == "neumann":
            lower[-1] -= 2.0 * alpha * self._inv_dr2
            main[-1] += 2.0 * alpha * self._inv_dr2
        elif self.bc_outer == "dirichlet":
            ri = r[-1]
            coef_2nd = alpha * self._inv_dr2
            coef_1st = alpha / (2.0 * dr * ri)
            lower[-1] -= (coef_2nd - coef_1st)
            main[-1] += 2.0 * coef_2nd + (coef_2nd + coef_1st)

        # 2. Reaction Decay Contribution (Coupled)
        # Add +0.5 * dt * K to the diagonal
        # This makes the matrix implicit for the reaction term too.
        if np.any(self.k != 0):
            reaction_term = 0.5 * self.dt * self.k
            main += reaction_term

        # Format for solve_banded (3, N)
        ab = np.zeros((3, N))
        ab[0, 1:] = upper[:-1]
        ab[1, :] = main
        ab[2, :-1] = lower[1:]
        return ab

    def _apply_explicit_operator(self, n, extra_source=None):
        """
        Apply (I + dt/2 * (D*Laplacian - K)) * n + dt * S
        """
        N = self.N
        dr = self.dr
        r = self.r
        alpha = self._alpha

        rhs = np.copy(n)

        # 1. Diffusion Operator
        # Center
        rhs[0] += 4.0 * alpha * (n[1] - n[0]) * self._inv_dr2

        # Interior
        if N > 2:
            ip = n[2:]
            i0 = n[1:-1]
            im = n[:-2]

            # Use precomputed coefficients
            coef_2nd = self._coef_2nd_interior
            coef_1st = self._coef_1st_interior

            lap = (ip - 2.0 * i0 + im) * coef_2nd
            lap += (ip - im) * coef_1st
            rhs[1:-1] += lap

        # Outer
        if self.bc_outer == "neumann":
            rhs[-1] += 2.0 * alpha * (n[-2] - n[-1]) * self._inv_dr2
        elif self.bc_outer == "dirichlet":
            ri = r[-1]
            coef_2nd = alpha * self._inv_dr2
            coef_1st = alpha / (2.0 * dr * ri)
            n_ghost = 2.0 * self.dirichlet_value - n[-1]
            lap = (n_ghost - 2.0 * n[-1] + n[-2]) * coef_2nd
            lap += (n_ghost - n[-2]) * coef_1st
            rhs[-1] += lap

        # 2. Reaction Decay Contribution
        # Subtract 0.5 * dt * K * n
        if np.any(self.k != 0):
            rhs -= 0.5 * self.dt * self.k * n

        # 3. Source Terms (Base + Extra)
        # Add full dt * S
        if np.any(self.s != 0):
            rhs += self.dt * self.s

        if extra_source is not None:
            rhs += self.dt * extra_source

        return rhs

    def step(self, n, dt=None, source=None):
        """
        Advance one time step.
        :param n: Current concentration
        :param dt: (Optional) New time step. If changed, matrix is rebuilt.
        :param source: (Optional) Additional time-dependent source term.
        """
        # Handle adaptive time stepping efficiently
        if dt is not None and dt != self.dt:
            self.set_dt(dt)

        # Compute Explicit Side (RHS)
        rhs = self._apply_explicit_operator(n, extra_source=source)

        # Solve Linear System (LHS)
        n_new = solve_banded((1, 1), self.ab, rhs)

        return n_new

    def compute_residual(self, n):
        """
        Compute the residual R(n) = D*Laplacian(n) - K*n + S.
        This represents ∂n/∂t at the current state.

        :param n: Current concentration array
        :return: Residual array (same shape as n)
        """
        N = self.N
        dr = self.dr
        r = self.r

        # We need the spatial derivative operator (Laplacian) independent of dt.
        # Laplacian = d2n/dr2 + (1/r)*dn/dr

        # 1. Diffusion Term: D * Laplacian(n)
        diff_term = np.zeros_like(n)
        inv_dr2 = self._inv_dr2

        # Interior points
        if N > 2:
            ip = n[2:]  # i+1
            i0 = n[1:-1]  # i
            im = n[:-2]  # i-1
            r_int = r[1:-1]

            # d2n/dr2
            d2n = (ip - 2.0 * i0 + im) * inv_dr2
            # (1/r) * dn/dr
            dn_r = (ip - im) / (2.0 * dr * r_int)

            diff_term[1:-1] = self.D * (d2n + dn_r)

        # Boundary: r = 0 (Symmetry)
        # Laplacian -> 4 * (n[1] - n[0]) / dr^2
        diff_term[0] = self.D * 4.0 * (n[1] - n[0]) * inv_dr2

        # Boundary: Outer (Neumann: dn/dr = 0 -> n_ghost = n_{N-2})
        if self.bc_outer == "neumann":
            # Laplacian -> 2 * (n[N-2] - n[N-1]) / dr^2
            diff_term[-1] = self.D * 2.0 * (n[-2] - n[-1]) * inv_dr2
        elif self.bc_outer == "dirichlet":
            # (Logic for Dirichlet if needed...)
            pass

        # 2. Reaction Decay: -K * n
        decay_term = -self.k * n if np.any(self.k != 0) else 0.0

        # 3. Source: +S
        source_term = self.s if np.any(self.s != 0) else 0.0

        # Total Residual = ∂n/∂t
        residual = diff_term + decay_term + source_term
        return residual

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
