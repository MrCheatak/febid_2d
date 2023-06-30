from processclass import Experiment2D
from analyse import get_peak
import program
import numpy as np

pr = Experiment2D()

pr.n0 = 2.7  # 1/nm^2
pr.F = 730.0  # 1/nm^2/s
pr.s = 1.0
pr.V = 0.05  # nm^3
pr.tau = 200e-6  # s
pr.D = 8e5  # nm^2/s
pr.sigma = 0.02  # nm^2
pr.f0 = 1.0e7
pr.fwhm = 50  # nm
pr.order = 1
pr.step = 0.5  # nm


if __name__ == '__main__':
    pr.tau_r
    pr.backend = 'gpu'
    pr.solve_steady_state()
    # r_max_vs_D()
    # r_max_vs_f0()
    # r_max_vs_tau()



