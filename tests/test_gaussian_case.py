import numpy as np
import pytest

from backend.diffusion import laplacian_radial_1d, laplace_1d, CrankNicolsonRadialSolver

# ==========
# Parameters
# ==========

D = 5          # diffusion coefficient
M = 1.0          # total mass
t_eq = 1 / (4 * np.pi *D)  # time at which analytical radial solution matches cartesian

t0 = 0.01 / D       # "regularization" time of the Gaussian at t=0
t1 = 0.05 / D       # scale by diffusion coefficient for generality and reasonable times
t2 = 0.1 / D
t3 = 0.2 / D        # largest test time

# Times at which we want to record profiles (rounded to nearest step)
record_times = np.array([t1, t2, t3])

# We choose R large such that Gaussian tail at r=R is negligible
# Require exp(-R^2 / (4 D (t0 + t3))) << 1. Take exponent ~ -20.
R = 8

N_r = 5000
r = np.linspace(0.0, R, N_r)
dr = r[1] - r[0]

# Stability limit for explicit Euler in radial 1D is conservatively
# dt <= dr^2 / (4 D)
dt = 0.2 * dr**2 / D   # safety factor

# Number of steps to reach t3 (we’ll integrate only once up to t3)
n_steps = int(np.round((t3) / dt))
t_effective = n_steps * dt  # actual max time we reach (close to t3)


# ============
# Helper funcs
# ============

def analytic_gaussian_radial(r, t, D_loc=None, M_loc=None, t0_loc=None):
    """
    Radial Green's function for 2D diffusion (cylindrical symmetry):
    n(r,t) = M / [4 π D (t0 + t)] * exp( - r^2 / [4 D (t0 + t)] )
    with a small t0 to regularize the initial delta.
    The solution scales with D*t, which can be used to test different D values.
    """
    if D_loc is None:
        D_loc = D
    if M_loc is None:
        M_loc = M
    if t0_loc is None:
        t0_loc = t0
    tau = t0_loc + t
    Dt = D_loc * tau  # diffusion time scale
    pref = M_loc / (4.0 * np.pi * Dt)
    return pref * np.exp(-r**2 / (4.0 * Dt))

def analytic_gaussian_cartesian(x, t, D_loc=None, M_loc=None, t0_loc=None):
    """
    Analytical solution for 1D Cartesian diffusion with a regularized
    point source at x = 0:
    u(x,t) = M / sqrt(4π D (t0 + t)) * exp( -x^2 / [4 D (t0 + t)] )
    with a small t0 to regularize the initial delta.
    """
    if D_loc is None:
        D_loc = D
    if M_loc is None:
        M_loc = M
    if t0_loc is None:
        t0_loc = t0
    tau = t0_loc + t
    Dt = D_loc * tau  # diffusion time scale
    pref = M_loc / np.sqrt(4.0 * np.pi * Dt)
    return pref * np.exp(-x**2 / (4.0 * Dt))


def compute_mass(n, r):
    """
    Compute total mass in cylindrical symmetry:
    M = ∫_0^R 2π r n(r) dr
    using the composite trapezoidal rule.
    """
    integrand = 2.0 * np.pi * r * n
    return np.trapz(integrand, r)


def cartesian_to_radial(u, t):
    """
    Transform a 1D Cartesian solution u(x, t) into a radial solution n(r, t),

    Parameters
    ----------
    u : ndarray, shape (N,)
        Cartesian solution values at positions r[i] (x ≥ 0). This should
        correspond to U(r, t) = 2π r n(r, t).

    Returns
    -------
    n : ndarray, shape (N,)
        Radial field n(r, t) consistent with the mapping above.
    """
    n = u / np.sqrt(4.0 * np.pi * D * (t0 + t))

    return n


def compute_r2_mean(n, r):
    """
    ⟨r^2⟩ = (∫ 2π r^3 n dr) / (∫ 2π r n dr)
    """
    numerator = np.sum(2.0 * np.pi * r**3 * n) * dr
    denominator = np.sum(2.0 * np.pi * r * n) * dr
    return numerator / denominator


def evolve_diffusion(n0, r, dr, dt, n_steps, D, record_times=record_times):
    """
    Simple explicit Euler evolution for radial diffusion:
    n^{k+1} = n^k + dt * D * Laplacian_radial(n^k)
    """
    n = n0.copy()
    snapshots = {}
    current_time = 0.0

    # Pre-compute which step indices correspond to desired record_times
    steps_for_record = {
        t_target: int(np.round(t_target / dt))
        for t_target in record_times
    }

    for step in range(n_steps + 1):
        # store snapshot if we are at one of the record times
        for t_target, step_target in steps_for_record.items():
            if step == step_target:
                snapshots[t_target] = n.copy()

        if step == n_steps:
            break

        lap = laplacian_radial_1d(n, r, dr, bc_outer="neumann")
        n = n + dt * D * lap
        current_time += dt

    return n, snapshots, current_time


def evolve_diffusion_cartesian(u0, x, dx, dt, n_steps, D, record_times=record_times):
    """
    Simple explicit Euler evolution for 1D Cartesian diffusion:
    u^{k+1} = u^k + dt * D * Laplacian_cartesian(u^k)

    Uses the laplace_1d operator from backend.diffusion module.

    Parameters
    ----------
    u0 : ndarray, shape (N,)
        Initial condition
    x : ndarray, shape (N,)
        Spatial grid points
    dx : float
        Grid spacing
    dt : float
        Time step
    n_steps : int
        Number of time steps to evolve
    D : float
        Diffusion coefficient
    record_times : list of float, optional
        Times at which to record snapshots. If None, uses the global record_times.

    Returns
    -------
    u_final : ndarray, shape (N,)
        Final state at t = n_steps * dt
    snapshots : dict
        Dictionary mapping record times to solution snapshots
    t_final : float
        Final time reached
    """
    u = u0.copy()
    snapshots = {}
    current_time = 0.0

    # Pre-compute which step indices correspond to desired record_times
    steps_for_record = {
        t_target: int(np.round(t_target / dt))
        for t_target in record_times
    }

    for step in range(n_steps + 1):
        # store snapshot if we are at one of the record times
        for t_target, step_target in steps_for_record.items():
            if step == step_target:
                snapshots[t_target] = u.copy()

        if step == n_steps:
            break

        # Use laplace_1d from diffusion module and scale by 1/dx^2
        lap = laplace_1d(u) / dx**2
        u = u + dt * D * lap
        current_time += dt

    return u, snapshots, current_time


# ========================
# 1) Mass conservation test
# ========================

def test_case2_mass_conservation_radial_fd():
    """
    Spreading Gaussian in semi-infinite domain test case.
    Check that total mass M is approximately conserved over time.
    """
    print("\nTesting radial FD mass conservation.")
    n0 = analytic_gaussian_radial(r, t=0.0)
    M0 = compute_mass(n0, r)

    # Check initial mass vs analytical
    print(f"M_analytical = {M}, M0_numerical = {M0:.6f}, rel_err = {abs(M0 - M) / M:.2e}")

    n_final, snapshots, t_final = evolve_diffusion(n0, r, dr, dt, n_steps, D)
    M_final = compute_mass(n_final, r)

    # Check how mass evolves
    print(f"M_final = {M_final:.6f}, rel_err = {abs(M_final - M) / M:.2e}")
    print(f"Mass change: {(M_final - M0) / M0 * 100:.2f}%")

    # Check boundary values
    print(f"n(r=0) initial: {n0[0]:.2e}, final: {n_final[0]:.2e}")
    print(f"n(r=R) initial: {n0[-1]:.2e}, final: {n_final[-1]:.2e}")
    # Initial condition at t=0: analytic Gaussian at t0
    n0 = analytic_gaussian_radial(r, t=0.0)

    M0 = compute_mass(n0, r)

    # Evolve
    n_final, snapshots, t_final = evolve_diffusion(
        n0, r, dr, dt, n_steps, D
    )

    M_final = compute_mass(n_final, r)

    # Analytically, total mass should stay = M.
    # Numerically, we accept a small relative error.
    rel_err_initial = abs(M0 - M) / M
    rel_err_final = abs(M_final - M) / M
    print(f"\n Radial FD mass conservation test:")
    print(f"  Initial mass error: {rel_err_initial:.2e}")
    print(f"  Final mass error: {rel_err_final:.2e}")

    # Tolerances – you can tighten once operator is confirmed correct
    assert rel_err_initial < 5e-4
    assert rel_err_final < 5e-3


# ===================================
# 2) Transient profile accuracy (L2/L∞)
# ===================================

@pytest.mark.parametrize("t_target", record_times)
def test_case2_transient_profiles_radial_fd(t_target):
    """
    Spreading Gaussian in semi-infinite domain test case.
    Compare numerical and analytical profiles from radial FD solver at t1, t2, t3 for accuracy.
    """
    print(f"\nTesting radial FD profile at t={t_target} for accuracy against analytical solution.")
    # Initial condition
    n0 = analytic_gaussian_radial(r, t=0.0)

    # Evolve once, reuse for all tests (cheap enough)
    n_final, snapshots, t_final = evolve_diffusion(
        n0, r, dr, dt, n_steps, D
    )

    assert t_target in snapshots, (
        f"No snapshot stored for t={t_target}, check record_times/dt."
    )

    n_num = snapshots[t_target]
    n_ana = analytic_gaussian_radial(r, t_target)

    # L2 error (area-weighted)
    diff = n_num - n_ana
    l2_num = np.sqrt(np.sum(2.0 * np.pi * r * diff**2) * dr)
    l2_ref = np.sqrt(np.sum(2.0 * np.pi * r * n_ana**2) * dr)
    rel_l2_err = l2_num / (l2_ref + 1e-15)

    # L∞ error
    linf_err = np.max(np.abs(diff))
    linf_ref = np.max(np.abs(n_ana))
    rel_linf_err = linf_err / (linf_ref + 1e-15)

    # Print diagnostics
    print(f"\nRadial FD profile test at t={t_target}:")
    print(f"  L2 relative error: {rel_l2_err:.2e}")
    print(f"  L∞ relative error: {rel_linf_err:.2e}")
    print(f"  n_ana[0]: {n_ana[0]:.4e}, n_num[0]: {n_num[0]:.4e}")

    # Tolerances (adjust as needed once operator & dt/dr are fixed)
    assert rel_l2_err < 5e-2
    assert rel_linf_err < 8e-2


@pytest.mark.parametrize("D", [1, 2, 5])
def test_case2_diffusion_coefficient_scan_radial_fd(D):
    """
    Spreading Gaussian in semi-infinite domain test case.
    Compare numerical and analytical profiles from radial FD solver at D1, D2 and D3 for accuracy.
    """
    print(f"\nTesting radial FD profile at t={t3 / D} and D={D} for accuracy against analytical solution.")
    t_0 = t0 / D  # rescaling regularization time
    t_target = t3 / D  # rescale target time
    dt = 0.2 * dr**2 / D  # rescale dt
    n_steps = int(np.round((t_target) / dt))  # rescale number of steps
    t_effective = n_steps * dt  # actual max time we reach (close to t3)
    # Initial condition
    n0 = analytic_gaussian_radial(r, t=0.0, D_loc=D, t0_loc=t_0)

    # Evolve once, reuse for all tests (cheap enough)
    n_final, snapshots, t_final = evolve_diffusion(
        n0, r, dr, dt, n_steps, D, record_times=[t_target]
    )

    assert t_target in snapshots, (
        f"No snapshot stored for t={t_target}, check record_times/dt."
    )

    n_num = snapshots[t_target]
    n_ana = analytic_gaussian_radial(r, t_target, t0_loc=t_0, D_loc=D)

    # L2 error (area-weighted)
    diff = n_num - n_ana
    l2_num = np.sqrt(np.sum(2.0 * np.pi * r * diff**2) * dr)
    l2_ref = np.sqrt(np.sum(2.0 * np.pi * r * n_ana**2) * dr)
    rel_l2_err = l2_num / (l2_ref + 1e-15)

    # L∞ error
    linf_err = np.max(np.abs(diff))
    linf_ref = np.max(np.abs(n_ana))
    rel_linf_err = linf_err / (linf_ref + 1e-15)

    # Print diagnostics
    print(f"\nRadial FD profile test at t={t_target} for D={D}")
    print(f"  L2 relative error: {rel_l2_err:.2e}")
    print(f"  L∞ relative error: {rel_linf_err:.2e}")
    print(f"  n_ana[0]: {n_ana[0]:.4e}, n_num[0]: {n_num[0]:.4e}")
    print(f"  n_ana[min]: {n_ana[np.argmin(n_ana)]:.4e}, n_num[min]: {n_num[np.argmin(n_num)]:.4e}")

    # Tolerances (adjust as needed once operator & dt/dr are fixed)
    assert rel_l2_err < 5e-2
    assert rel_linf_err < 8e-2


@pytest.mark.parametrize("D", [1, 2, 5])
def test_case2_diffusion_coefficient_scan_cartesian_fd(D):
    """
    Spreading Gaussian in semi-infinite domain test case.
    Compare numerical and analytical profiles of 1D Cartesian FD solver at D1, D2 and D3.
    """
    print(f"\nTesting Cartesian FD profile at t={t3 / D} and D={D} for accuracy against analytical solution.")
    t_0 = t0 / D
    t_target = t3 / D
    dt = 0.2 * dr**2 / D
    n_steps = int(np.round((t_target) / dt))
    t_effective = n_steps * dt  # actual max time we reach (close to t3)
    # Initial condition
    n0 = analytic_gaussian_cartesian(r, t=0.0, D_loc=D, t0_loc=t_0)

    # Evolve once, reuse for all tests (cheap enough)
    n_final, snapshots, t_final = evolve_diffusion_cartesian(
        n0, r, dr, dt, n_steps, D, record_times=[t_target]
    )

    assert t_target in snapshots, (
        f"No snapshot stored for t={t_target}, check record_times/dt."
    )

    n_num = snapshots[t_target]
    n_ana = analytic_gaussian_cartesian(r, t_target, t0_loc=t_0, D_loc=D)

    # L2 error (area-weighted)
    diff = n_num - n_ana
    l2_num = np.sqrt(np.sum(2.0 * np.pi * r * diff**2) * dr)
    l2_ref = np.sqrt(np.sum(2.0 * np.pi * r * n_ana**2) * dr)
    rel_l2_err = l2_num / (l2_ref + 1e-15)

    # L∞ error
    linf_err = np.max(np.abs(diff))
    linf_ref = np.max(np.abs(n_ana))
    rel_linf_err = linf_err / (linf_ref + 1e-15)

    # Print diagnostics
    print(f"\nCartesian FD profile test at t={t_target} for D={D}")
    print(f"  L2 relative error: {rel_l2_err:.2e}")
    print(f"  L∞ relative error: {rel_linf_err:.2e}")
    print(f"  n_ana[0]: {n_ana[0]:.4e}, n_num[0]: {n_num[0]:.4e}")
    print(f"  n_ana[min]: {n_ana[np.argmin(n_ana)]:.4e}, n_num[min]: {n_num[np.argmin(n_num)]:.4e}")

    # Tolerances (adjust as needed once operator & dt/dr are fixed)
    assert rel_l2_err < 5e-2
    assert rel_linf_err < 8e-2

# ========================================
# 3) <r^2> vs time: variance growth ~ 4Dt
# ========================================

@pytest.mark.parametrize("t_target", record_times)
def test_case2_variance_growth_radial_fd(t_target):
    """
    Spreading Gaussian in semi-infinite domain test case.
    The analytical solution has
        <r^2>(t) = 4 D (t0 + t)
    Check that numerical <r^2>(t) follows this law at t1, t2, t3.
    """
    print(f"\nTesting radial FD variance growth at t={t_target}.")
    n0 = analytic_gaussian_radial(r, t=0.0)
    n_final, snapshots, t_final = evolve_diffusion(
        n0, r, dr, dt, n_steps, D
    )
    n_num = snapshots[t_target]

    r2_num = compute_r2_mean(n_num, r)
    r2_ana = 4.0 * D * (t0 + t_target)

    rel_err = abs(r2_num - r2_ana) / (r2_ana + 1e-15)
    print(f"\nRadial FD variance growth at t={t_target}:")
    print(f"  <r^2>_num = {r2_num:.4f}, <r^2>_ana = {r2_ana:.4f}, rel_err = {rel_err:.2e}")

    # Again, you can tighten later
    assert rel_err < 5e-2


# ========================================
# 4) Cartesian-to-radial transformation consistency
# ========================================

@pytest.mark.parametrize("t_target", record_times)
def test_cartesian_radial_fd_equivalence(t_target):
    """
    Case 2: Verify that the Cartesian numerical solution, when transformed
    to radial coordinates via cartesian_to_radial(), matches the native
    radial numerical solution.

    This tests that when both solutions start from equivalent initial conditions
    and are evolved with their respective operators, the Cartesian solution
    can be transformed back to match the radial solution.

    Transformation:
    - Forward: U0 = n0 * sqrt(4πD(t0 + 0))
    - Backward: n = cartesian_to_radial(U, t)
    """
    print(f"\nTesting Cartesian-to-radial FD equivalence at t={t_target}.")
    # Evolve radial solution
    n0_radial = analytic_gaussian_radial(r, t=0.0)
    _, snapshots_radial, _ = evolve_diffusion(n0_radial, r, dr, dt, n_steps, D)

    # Evolve Cartesian solution with transformed initial condition
    # U0 = n0 * sqrt(4πD(t0 + 0))
    U0 = n0_radial * np.sqrt(4.0 * np.pi * D * (t0 + 0.0))
    _, snapshots_cart, _ = evolve_diffusion_cartesian(
        U0, r, dr, dt, n_steps, D
    )

    # Get solutions at target time
    assert t_target in snapshots_radial, f"No radial snapshot for t={t_target}"
    assert t_target in snapshots_cart, f"No Cartesian snapshot for t={t_target}"

    n_radial = snapshots_radial[t_target]
    U_cart = snapshots_cart[t_target]

    # Transform Cartesian solution back to radial
    n_from_cart = cartesian_to_radial(U_cart, t_target)

    # Compare the two radial solutions
    diff = n_radial - n_from_cart

    # L2 error (area-weighted in radial coordinates)
    l2_num = np.sqrt(np.sum(2.0 * np.pi * r * diff**2) * dr)
    l2_ref = np.sqrt(np.sum(2.0 * np.pi * r * n_radial**2) * dr)
    rel_l2_err = l2_num / (l2_ref + 1e-15)

    # L∞ error
    linf_err = np.max(np.abs(diff))
    linf_ref = np.max(np.abs(n_radial))
    rel_linf_err = linf_err / (linf_ref + 1e-15)

    # Print diagnostics
    print(f"\nCartesian→radial equivalence test at t={t_target}:")
    print(f"  L2 relative error: {rel_l2_err:.2e}")
    print(f"  L∞ relative error: {rel_linf_err:.2e}")
    print(f"  n_radial[0]: {n_radial[0]:.4e}, n_from_cart[0]: {n_from_cart[0]:.4e}")
    print(f"  n_radial[max]: {n_radial[np.argmax(n_radial)]:.4e}")
    print(f"  Diff at r=0: {diff[0]:.4e}")

    # The two solutions should match well if the transformation is correct
    # and both numerical schemes are consistent
    assert rel_l2_err < 0.05, f"L2 error too large: {rel_l2_err:.2e}"
    assert rel_linf_err < 0.1, f"L∞ error too large: {rel_linf_err:.2e}"


# ========================================
# 5) Crank-Nicolson solver tests
# ========================================

def test_radial_crank_nicolson_mass_conservation():
    """
    Test that the radial Crank-Nicolson solver conserves mass over time.
    """
    print("\nTesting radial Crank-Nicolson mass conservation.")
    # Initial condition
    n0 = analytic_gaussian_radial(r, t=0.0)
    M0 = compute_mass(n0, r)

    # Use larger time step (CN is unconditionally stable)
    dt_cn = 1000 * dt  # 5x larger than explicit Euler
    n_steps_cn = int(np.round(t3 / dt_cn))

    # Initialize and run CN solver
    solver = CrankNicolsonRadialSolver(r, D, dt_cn, bc_outer="neumann")

    # Evolve to final time
    n = n0.copy()
    for i in range(n_steps_cn):
        n = solver.step(n)

    M_final = compute_mass(n, r)

    # Mass should be conserved extremely well with CN
    rel_err_initial = abs(M0 - M) / M
    rel_err_final = abs(M_final - M) / M

    print(f"\nCrank-Nicolson mass conservation:")
    print(f"  Initial mass error: {rel_err_initial:.2e}")
    print(f"  Final mass error: {rel_err_final:.2e}")
    print(f"  Mass change: {(M_final - M0) / M0 * 100:.4f}%")

    # CN should conserve mass better than explicit Euler
    assert rel_err_initial < 5e-4
    assert rel_err_final < 1e-2  # Should be much better than explicit


@pytest.mark.parametrize("t_target", record_times)
def test_radial_crank_nicolson_accuracy(t_target):
    """
    Test radial Crank-Nicolson solver accuracy against analytical solution.

    CN is second-order accurate in time, so it should match the analytical
    solution better than first-order explicit Euler, especially for larger
    time steps.
    """
    print(f"\nTesting radial Crank-Nicolson profile at t={t_target} for accuracy against analytical solution.")
    # Initial condition
    n0 = analytic_gaussian_radial(r, t=0.0)

    # Use larger time step to demonstrate CN stability and accuracy
    dt_cn = 10 * dt

    # Initialize solver
    solver = CrankNicolsonRadialSolver(r, D, dt_cn, bc_outer="neumann")

    # Solve using the solve method
    times, solutions = solver.solve(n0, t_target, store_every=1)

    # Get solution at target time (last stored solution)
    n_num = solutions[-1, :]
    n_ana = analytic_gaussian_radial(r, t_target)

    # L2 error (area-weighted)
    diff = n_num - n_ana
    l2_num = np.sqrt(np.sum(2.0 * np.pi * r * diff**2) * dr)
    l2_ref = np.sqrt(np.sum(2.0 * np.pi * r * n_ana**2) * dr)
    rel_l2_err = l2_num / (l2_ref + 1e-15)

    # L∞ error
    linf_err = np.max(np.abs(diff))
    linf_ref = np.max(np.abs(n_ana))
    rel_linf_err = linf_err / (linf_ref + 1e-15)

    print(f"\nCrank-Nicolson accuracy test at t={t_target}:")
    print(f"  L2 relative error: {rel_l2_err:.2e}")
    print(f"  L∞ relative error: {rel_linf_err:.2e}")
    print(f"  dt_CN / dt_explicit = {dt_cn / dt:.1f}")

    # CN should be accurate even with larger time steps
    # These tolerances should be comparable or better than explicit Euler
    assert rel_l2_err < 5e-2, f"L2 error too large: {rel_l2_err:.2e}"
    assert rel_linf_err < 8e-2, f"L∞ error too large: {rel_linf_err:.2e}"


@pytest.mark.parametrize("D", [1, 2, 5])
def test_diffusion_coefficient_scan_radial_crank_nicholson(D):
    """
    Test radial Crank-Nicolson solver accuracy against analytical solution at D1, D2, D3.

    CN is second-order accurate in time, so it should match the analytical
    solution better than first-order explicit Euler, especially for larger
    time steps.
    """
    t_0 = t0 / D
    t_target = t3 / D
    dt = 0.2 * dr ** 2 / D
    n_steps = int(np.round((t3) / dt))
    t_effective = n_steps * dt  # actual max time we reach (close to t3)
    # Initial condition
    n0 = analytic_gaussian_radial(r, t=0.0, D_loc=D, t0_loc=t_0)

    # Use larger time step to demonstrate CN stability and accuracy
    dt_cn = 25 * dt

    # Initialize solver
    solver = CrankNicolsonRadialSolver(r, D, dt_cn, bc_outer="neumann")

    # Solve using the solve method
    times, solutions = solver.solve(n0, t_target, store_every=1)

    # Get solution at target time (last stored solution)
    n_num = solutions[-1, :]
    n_ana = analytic_gaussian_radial(r, t_target, D_loc=D, t0_loc=t_0)

    # L2 error (area-weighted)
    diff = n_num - n_ana
    l2_num = np.sqrt(np.sum(2.0 * np.pi * r * diff**2) * dr)
    l2_ref = np.sqrt(np.sum(2.0 * np.pi * r * n_ana**2) * dr)
    rel_l2_err = l2_num / (l2_ref + 1e-15)

    # L∞ error
    linf_err = np.max(np.abs(diff))
    linf_ref = np.max(np.abs(n_ana))
    rel_linf_err = linf_err / (linf_ref + 1e-15)

    print(f"\nCrank-Nicolson accuracy test at t={t_target} for D={D}")
    print(f"  L2 relative error: {rel_l2_err:.2e}")
    print(f"  L∞ relative error: {rel_linf_err:.2e}")
    print(f"  n_ana[0]: {n_ana[0]:.4e}, n_num[0]: {n_num[0]:.4e}")
    print(f"  n_ana[min]: {n_ana[np.argmin(n_ana)]:.4e}, n_num[min]: {n_num[np.argmin(n_num)]:.4e}")
    print(f"  dt_CN / dt_explicit = {dt_cn / dt:.1f}")

    # CN should be accurate even with larger time steps
    # These tolerances should be comparable or better than explicit Euler
    assert rel_l2_err < 5e-2, f"L2 error too large: {rel_l2_err:.2e}"
    assert rel_linf_err < 8e-2, f"L∞ error too large: {rel_linf_err:.2e}"


def test_crank_nicolson_variance_growth():
    """
    Test that CN solver correctly reproduces the variance growth <r^2>(t) = 4Dt.

    This tests the overall diffusive behavior of the scheme.
    """
    print("\nTesting Crank-Nicolson variance growth.")
    # Initial condition
    n0 = analytic_gaussian_radial(r, t=0.0)

    # Use moderately large time step
    dt_cn = 25 * dt

    solver = CrankNicolsonRadialSolver(r, D, dt_cn, bc_outer="neumann")

    # Test at each record time
    for t_target in record_times:
        n_steps_to_target = int(np.round(t_target / dt_cn))

        # Evolve
        n = n0.copy()
        for _ in range(n_steps_to_target):
            n = solver.step(n)

        r2_num = compute_r2_mean(n, r)
        r2_ana = 4.0 * D * (t0 + t_target)

        rel_err = abs(r2_num - r2_ana) / (r2_ana + 1e-15)

        print(f"\nCN variance at t={t_target}: num={r2_num:.4f}, ana={r2_ana:.4f}, rel_err={rel_err:.2e}")

        assert rel_err < 5e-2, f"Variance error too large at t={t_target}: {rel_err:.2e}"


def test_crank_nicolson_vs_explicit_euler():
    """
    Compare Crank-Nicolson with explicit Euler at the same effective resolution.

    CN should give comparable or better accuracy with larger time steps,
    demonstrating its superior stability properties.
    """
    print("\nComparing Crank-Nicolson vs Explicit Euler accuracy.")
    # Initial condition
    n0 = analytic_gaussian_radial(r, t=0.0)
    t_target = t3  # Test at intermediate time

    # Explicit Euler (small time step)
    n_euler, snapshots_euler, _ = evolve_diffusion(n0, r, dr, dt, n_steps, D)
    n_euler_at_t2 = snapshots_euler[t_target]

    # Crank-Nicolson (larger time step)
    dt_cn = 25 * dt
    n_steps_cn = int(np.round(t_target / dt_cn))

    solver = CrankNicolsonRadialSolver(r, D, dt_cn, bc_outer="neumann")
    n_cn = n0.copy()
    for _ in range(n_steps_cn):
        n_cn = solver.step(n_cn)

    # Analytical solution
    n_ana = analytic_gaussian_radial(r, t_target)

    # Compare errors
    def compute_rel_l2_error(n_num, n_ref):
        diff = n_num - n_ref
        l2_num = np.sqrt(np.sum(2.0 * np.pi * r * diff**2) * dr)
        l2_ref = np.sqrt(np.sum(2.0 * np.pi * r * n_ref**2) * dr)
        return l2_num / (l2_ref + 1e-15)

    err_euler = compute_rel_l2_error(n_euler_at_t2, n_ana)
    err_cn = compute_rel_l2_error(n_cn, n_ana)

    print(f"\nExplicit Euler vs Crank-Nicolson comparison at t={t_target}:")
    print(f"  Euler error (dt={dt:.2e}):  {err_euler:.2e}")
    print(f"  CN error (dt={dt_cn:.2e}):     {err_cn:.2e}")
    print(f"  CN uses {dt_cn/dt:.1f}x larger time step")

    # CN should be at least as accurate as Euler despite larger time step
    # This demonstrates the superior stability and accuracy of the implicit scheme
    assert err_cn < 2 * err_euler, "CN should be competitive with Euler even at larger dt"


