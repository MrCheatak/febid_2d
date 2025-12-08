import sys
import os
from timeit import default_timer as dtm
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.processclass import Experiment1D
import numpy as np
import matplotlib.pyplot as plt


print("Initializing precursor parameters...")
pr = Experiment1D()
pr.n0 = 2.8  # 1/nm^2
pr.F = 2300  # 1/nm^2/s
pr.s = 0.025
pr.V = 8.41e-3  # nm^3
pr.tau = 1.2e-3  # s
pr.D = 1.4e6  # nm^2/s
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

def test_cases_cn_vs_fd():
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
    start_time = dtm()
    n_cn = pr.solve_steady_state(n_init=n_init, progress=True, plot_fit=True, init_tol=1e-5)
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


if __name__ == "__main__":
    test_cases_cn_vs_fd()