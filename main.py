from processclass import Experiment2D
from timeit import default_timer as dt

pr = Experiment2D()

pr.n0 = 2.7  # 1/nm^2
pr.F = 730.0  # 1/nm^2/s
pr.s = 0.0062
pr.V = 0.05  # nm^3
pr.tau = 40000e-6  # s
pr.D = 9e6  # nm^2/s
pr.sigma = 0.002  # nm^2
pr.f0 = 6.3e6
pr.fwhm = 1000  # nm
pr.order = 1
pr.step = 1   # nm


if __name__ == '__main__':
    pr.tau_r
    pr.backend = 'gpu'
    pr.solve_steady_state(eps=1e-7, progress=True)
    pr.plot('R')
    pr.backend = 'gpu'
    # t1 = dt()
    # pr.solve_steady_state(eps=1e-8, progress=True)
    # t2 = dt() - t1
    # print(f'GPU took {t2:.3f} s')
    # pr.plot('R')
    # pr.backend = 'cpu'
    # t1 = dt()
    # pr.solve_steady_state(eps=1e-8, progress=True)
    # t2 = dt() - t1
    # print(f'CPU took {t2:.3f} s')
    # pr.plot('R')
    # r_max_vs_D()
    # r_max_vs_f0()
    # r_max_vs_tau()



