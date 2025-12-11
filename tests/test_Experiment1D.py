import sys
import os
import pytest
import numpy as np
from copy import deepcopy
from timeit import default_timer as dtm
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve

from backend.diffusion import CrankNicolsonRadialSolver
from backend.processclass import Experiment1D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


print("Initializing precursor parameters...")
pr = Experiment1D()
pr.n0 = 2.8  # 1/nm^2
pr.F = 2300  # 1/nm^2/s
pr.s = 0.025
pr.V = 8.41e-3  # nm^3
pr.tau = 1.2e-3  # s
pr.D = 3.4e6  # nm^2/s
pr.sigma = 0.065  # nm^2
pr.st_dev = 3  # nm
pr.f0 = 9.8e5
pr.step = pr.fwhm /50  # nm
pr.beam_type = 'gauss'
pr.order = 4
pr.coords = 'radial'
pr.num_scheme = 'fd'
dt = pr.dt
pr.get_grid()
n_init = np.zeros_like(pr.r)
n_init.fill(pr.nr)

def test_case_cn_vs_fd():
    print(f"dt_diss: {pr.dt_diss:.4e}, dt_des: {pr.dt_des:.4e}, dt_diff: {pr.dt_diff:.6e}")
    print(f"dt_fd: {pr.dt_fd:.3e}")
    print(f"Running FD scheme...")
    print(f"Grid size: {len(pr.r)}, dt: {pr.dt}, D: {pr.D}")
    start_time = dtm()
    n_fd = pr.solve_steady_state(n_init=n_init, progress=True, plot_fit=True)
    end_time = dtm()
    print(f"FD solution completed. Min n: {n_fd.min():.4f}")
    print(f"Relaxation time: {pr._tr:.2e} seconds")

    print(f"Time taken for FD: {end_time - start_time:.2f} seconds")

    print(f"\nRunning CN scheme...")
    pr.num_scheme = 'cn'
    pr.cn_dt_max_factor = 2000
    start_time = dtm()
    n_cn = pr.solve_steady_state(n_init=n_init, progress=True, verbose=True, plot_fit=False, init_tol=1e-5)
    end_time = dtm()
    print(f"CN solution completed. Min n: {n_cn.min():.4f}")
    print(f"Relaxation time: {pr._tr:.2e} seconds")
    print(f"Time taken for CN: {end_time - start_time:.2f} seconds")
    print(f"dt_cn: {pr.dt_cn:.3e}")
    print(f"  CN uses {pr.dt_cn / pr.dt_fd:.1f}x larger time step")

    # Compute errors
    diff = n_fd - n_cn

    # L2 error (area-weighted in radial coordinates)
    r = pr.r
    dr = pr.step
    l2_num = np.sqrt(np.sum(2.0 * np.pi * r * diff ** 2) * dr)
    l2_ref = np.sqrt(np.sum(2.0 * np.pi * r * n_fd ** 2) * dr)
    rel_l2_err = l2_num / (l2_ref + 1e-15)

    # L∞ error
    linf_err = np.max(np.abs(diff))
    linf_ref = np.max(np.abs(n_fd))
    rel_linf_err = linf_err / (linf_ref + 1e-15)

    print(f"  L2 relative error: {rel_l2_err:.2e}")
    print(f"  L∞ relative error: {rel_linf_err:.2e}")
    print(f"  n_radial[0]: {n_fd[0]:.4e}, n_cn[0]: {n_cn[0]:.4e}")
    print(f"  n_radial[max]: {n_fd[np.argmax(n_fd)]:.4e}")
    print(f"  n_cn[max]: {n_cn[np.argmax(n_cn)]:.4e}")
    print(f"  Diff at r=0: {diff[0]:.4e}")


    # The two solutions should match well if both numerical schemes are consistent
    assert rel_l2_err < 0.05, f"L2 error too large: {rel_l2_err:.2e}"
    assert rel_linf_err < 0.1, f"L∞ error too large: {rel_linf_err:.2e}"

def test_solution_trueness_crank_nicolson(pr=pr):
    """
    Test that the Crank-Nicolson numerical solution closely matches the analytical solution for a small D case.
    :return:
    """
    print("\nTesting solution trueness for small D case...")
    pr1 = deepcopy(pr)
    pr = pr1
    pr.D = 1e2  # nm^2/s
    r = pr.get_grid()
    n_init = np.zeros_like(r)
    n_init.fill(pr.nr)

    print(f"Running CN scheme for small D...")
    pr.num_scheme = 'cn'
    pr.cn_dt_max_factor = 500
    n_cn = pr.solve_steady_state(n_init=n_init, progress=True, verbose=True, plot_fit=False, init_tol=1e-5)

    # Analytical solution (neglecting diffusion)
    n_analytical = pr.analytic(r)

    # Compute errors
    diff = n_analytical - n_cn

    # L2 error (area-weighted in radial coordinates)
    dr = pr.step
    l2_num = np.sqrt(np.sum(2.0 * np.pi * r * diff ** 2) * dr)
    l2_ref = np.sqrt(np.sum(2.0 * np.pi * r * n_analytical ** 2) * dr)
    rel_l2_err = l2_num / (l2_ref + 1e-15)

    # L∞ error
    linf_err = np.max(np.abs(diff))
    linf_ref = np.max(np.abs(n_analytical))
    rel_linf_err = linf_err / (linf_ref + 1e-15)

    print(f"n_analytical[0]: {n_analytical[0]:.4e}, n_cn[0]: {n_cn[0]:.4e}")
    print(f"n_analytical[max]: {n_analytical[np.argmax(n_analytical)]:.4e}, n_cn[max]: {n_cn[np.argmax(n_cn)]:.4e}")
    print(f"Diff at r=0: {diff[0]:.4e}")
    print(f"  L2 relative error: {rel_l2_err:.2e}")
    print(f"  L∞ relative error: {rel_linf_err:.2e}")

    # The numerical solution should closely match the analytical solution for small D
    assert rel_l2_err < 0.02, f"L2 error too large for small D case: {rel_l2_err:.2e}"
    assert rel_linf_err < 0.03, f"L∞ error too large for small D case: {rel_linf_err:.2e}"


def test_solution_trueness_finite_difference(pr=pr):
    """
    Test that the explicit Euler numerical solution closely matches the analytical solution for a small D case.
    :return:
    """
    print("\nTesting solution trueness for small D case...")
    pr1 = deepcopy(pr)
    pr = pr1
    pr.D = 1e2  # nm^2/s
    r = pr.get_grid()
    n_init = np.zeros_like(r)
    n_init.fill(pr.nr)

    print(f"Running FD scheme for small D...")
    pr.num_scheme = 'fd'
    n_fd = pr.solve_steady_state(n_init=n_init, progress=True, verbose=True, plot_fit=False)

    # Analytical solution (neglecting diffusion)

    r = pr.get_grid()
    n_analytical = pr.analytic(r)

    # Compute errors
    diff = n_analytical - n_fd

    # L2 error (area-weighted in radial coordinates)
    dr = pr.step
    l2_num = np.sqrt(np.sum(2.0 * np.pi * r * diff ** 2) * dr)
    l2_ref = np.sqrt(np.sum(2.0 * np.pi * r * n_analytical ** 2) * dr)
    rel_l2_err = l2_num / (l2_ref + 1e-15)

    # L∞ error
    linf_err = np.max(np.abs(diff))
    linf_ref = np.max(np.abs(n_analytical))
    rel_linf_err = linf_err / (linf_ref + 1e-15)

    print(f"n_analytical[0]: {n_analytical[0]:.4e}, n_fd[0]: {n_fd[0]:.4e}")
    print(f"n_analytical[max]: {n_analytical[np.argmax(n_analytical)]:.4e}, n_fd[max]: {n_fd[np.argmax(n_fd)]:.4e}")
    print(f"Diff at r=0: {diff[0]:.4e}")
    print(f"  L2 relative error: {rel_l2_err:.2e}")
    print(f"  L∞ relative error: {rel_linf_err:.2e}")

    # The numerical solution should closely match the analytical solution for small D
    assert rel_l2_err < 0.02, f"L2 error too large for small D case: {rel_l2_err:.2e}"
    assert rel_linf_err < 0.03, f"L∞ error too large for small D case: {rel_linf_err:.2e}"

@pytest.mark.parametrize("dt_factor", [10000, 5000, 1000, 500])
def test_timestep_factor_cn(dt_factor, pr=pr):
    """
    Test various time step factors for the CN scheme and compare solutions and relaxation times
    :return:
    """
    print("\nTesting time step factor scaling for CN case...")
    pr1 = deepcopy(pr)
    pr = pr1
    pr.D = 6.4e6
    pr.tau = 3.6e-3
    pr.fwhm = 3
    pr.f0 = 1.8e8
    pr.cn_dt_max_factor = dt_factor
    r = pr.get_grid()
    n_init = np.zeros_like(r)
    n_init.fill(pr.nr)

    print(f"Running FD scheme for small D...")
    pr.num_scheme = 'cn'
    n_cn = pr.solve_steady_state(n_init=n_init, progress=True, verbose=True, plot_fit=False)

    print(f" n_cn[0]: {n_cn[0]:.4e}")
    print(f"n_cn[max]: {n_cn[np.argmax(n_cn)]:.4e}")
    print(f"Relaxation time: {pr._tr:.6f} seconds with dt factor: {dt_factor}")


def test_boundary_valve_problem_fd(pr=pr):
    """
    Test direct BVP solution against time-stepped result.
    :param pr:
    :return:
    """
    print("\nTesting solution against BVP solution...")
    pr1 = deepcopy(pr)
    pr = pr1
    # pr.D = 6.4e6
    # pr.tau = 3.6e-3
    # pr.fwhm = 1.5
    # pr.f0 = 1.8e9
    r = pr.get_grid()
    n_init = np.zeros_like(r)
    n_init.fill(pr.nr)

    print(f"Running FD scheme...")
    pr.num_scheme = 'fd'
    n_fd = pr.solve_steady_state(n_init=n_init, depletion=None, progress=True, plot_fit=False)

    print(f"FD solution completed. Min n: {n_fd.min():.4f}")
    print(f"Relaxation time: {pr._tr:.2e} seconds")

    n_ss = pr.get_steady_state()
    print(f"BVP solution completed. Min n: {n_ss.min():.4f}")

    # Compare
    diff = n_ss - n_fd

    # L2 error (area-weighted in radial coordinates)
    dr = pr.step
    l2_num = np.sqrt(np.sum(2.0 * np.pi * r * diff ** 2) * dr)
    l2_ref = np.sqrt(np.sum(2.0 * np.pi * r * n_ss ** 2) * dr)
    rel_l2_err = l2_num / (l2_ref + 1e-15)

    # L∞ error
    linf_err = np.max(np.abs(diff))
    linf_ref = np.max(np.abs(n_ss))
    rel_linf_err = linf_err / (linf_ref + 1e-15)

    print(f"n_ss[0]: {n_ss[0]:.4e}, n_fd[0]: {n_fd[0]:.4e}")
    print(f"n_ss[max]: {n_ss[np.argmax(n_ss)]:.4e}, n_fd[max]: {n_fd[np.argmax(n_fd)]:.4e}")
    print(f"Diff at r=0: {diff[0]:.4e}")
    print(f"  L2 relative error: {rel_l2_err:.2e}")
    print(f"  L∞ relative error: {rel_linf_err:.2e}")

    # The numerical solution should closely match the analytical solution for small D
    assert rel_l2_err < 0.02, f"L2 error too large for small D case: {rel_l2_err:.2e}"
    assert rel_linf_err < 0.03, f"L∞ error too large for small D case: {rel_linf_err:.2e}"




def test_boundary_valve_problem_cn(pr=pr):
    """
    Compare time-stepped result against direct BVP solution.
    Solves: D*Laplacian(n) - K*n = -S
    """
    # 1. Build the operator matrix L explicitly
    # We can reuse the solver's internal structure but need to strip the 'dt' dependence
    # Or just construct a sparse matrix for D*Lapl - K
    print("\nTesting solution against BVP solution...")
    pr1 = deepcopy(pr)
    pr = pr1
    pr.D = 6.4e6
    pr.tau = 3.6e-3
    pr.fwhm = 1.5
    pr.f0 = 1.8e9
    pr.cn_dt_max_factor = 100000
    r = pr.get_grid()
    n_init = np.zeros_like(r)
    n_init.fill(pr.nr)

    print(f"Running CN scheme...")
    pr.num_scheme = 'cn'
    n_final = pr.solve_steady_state(n_init=n_init, progress=True, verbose=True, plot_fit=False)
    print(f"CN solution completed. Min n: {n_final.min():.4f}")
    print(f"Relaxation time: {pr._tr:.2e} seconds")
    reaction_k = pr.s * pr.F / pr.n0 + 1 / pr.tau + pr.sigma * pr.f
    reaction_s = pr.s * pr.F
    dt = pr.dt * pr.cn_dt_max_factor
    # Initialize solver with reaction terms
    solver = CrankNicolsonRadialSolver(
        r=r,
        D=pr.D,
        dt=dt,
        reaction_k=reaction_k,
        reaction_s=reaction_s,
        bc_outer="neumann"
    )
    N = solver.N
    dr = solver.dr
    r = solver.r
    D = solver.D
    K = solver.k if isinstance(solver.k, np.ndarray) else np.full(N, solver.k)
    S = solver.s if isinstance(solver.s, np.ndarray) else np.full(N, solver.s)

    # Build Tridiagonal Matrix for L = D*(d2/dr2 + 1/r d/dr) - K
    diagonals = [np.zeros(N), np.zeros(N), np.zeros(N)]  # Lower, Main, Upper

    inv_dr2 = 1.0 / (dr ** 2)

    # Interior Points
    for i in range(1, N - 1):
        # Coeffs for: D * (n[i+1] - 2n[i] + n[i-1])/dr^2
        c2 = D * inv_dr2
        # Coeffs for: D * (1/r) * (n[i+1] - n[i-1])/(2dr)
        c1 = D / (2 * r[i] * dr)

        diagonals[0][i - 1] = c2 - c1  # Lower (coefficient of i-1)
        diagonals[1][i] = -2 * c2 - K[i]  # Main  (coefficient of i)
        diagonals[2][i + 1] = c2 + c1  # Upper (coefficient of i+1)

    # BC: r=0 (Symmetry) -> 4 * D * (n[1]-n[0])/dr^2 - K[0]*n[0]
    diagonals[1][0] = -4 * D * inv_dr2 - K[0]
    diagonals[2][1] = 4 * D * inv_dr2

    # BC: Outer (Neumann) -> 2 * D * (n[N-2]-n[N-1])/dr^2 - K[-1]*n[-1]
    diagonals[0][N - 2] = 2 * D * inv_dr2
    diagonals[1][N - 1] = -2 * D * inv_dr2 - K[-1]

    # Construct exact arrays for diags
    # Lower diagonal (offset -1): Entry k corresponds to A[k+1, k]
    # We stored A[i, i-1] at index i-1.
    # Letting k = i-1, we need index k. So we take the sequence from 0.
    lower_data = diagonals[0][:-1]  # Cut off the last unused element

    # Main diagonal (offset 0)
    main_data = diagonals[1]

    # Upper diagonal (offset +1): Entry k corresponds to A[k, k+1]
    # We stored A[i, i+1] at index i+1.
    # Letting k = i, we need index k+1. So we shift left by 1.
    upper_data = diagonals[2][1:]   # Cut off the first unused element

    # Construct sparse matrix
    matrix = diags(
        [lower_data, main_data, upper_data],
        [-1, 0, 1], shape=(N, N), format='csc'
    )

    # Solve L*n = -S
    rhs = -S
    n_steady_bvp = spsolve(matrix, rhs)
    print(f"BVP solution completed. Min n: {n_steady_bvp.min():.4f}")

    # Compare
    diff = np.linalg.norm(n_final - n_steady_bvp) / np.linalg.norm(n_steady_bvp)
    print(f"Difference between Time-Stepper and Exact BVP: {diff:.3e}")
    assert diff < 1e-5


if __name__ == "__main__":
    test_case_cn_vs_fd()